#include "gpu_tests.h"
#include "shaders.h"

#include <libplacebo/renderer.h>
#include <libplacebo/shaders/dithering.h>
#include <libplacebo/shaders/motion.h>
#include <libplacebo/utils/frame_queue.h>
#include <libplacebo/utils/upload.h>

#include "dispatch.h"

//#define PRINT_OUTPUT

void pl_buffer_tests(pl_gpu gpu)
{
    const size_t buf_size = 1024;
    if (buf_size > gpu->limits.max_buf_size)
        return;
    printf("pl_buffer_tests:\n");

    uint8_t *test_src = malloc(buf_size * 2);
    uint8_t *test_dst = test_src + buf_size;
    assert(test_src && test_dst);
    memset(test_dst, 0, buf_size);
    for (int i = 0; i < buf_size; i++)
        test_src[i] = RANDOM_U8;

    pl_buf buf = NULL, tbuf = NULL;

    printf("- test buffer static creation and readback\n");
    buf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .host_readable = true,
        .initial_data = test_src,
    ));

    REQUIRE(buf);
    REQUIRE(pl_buf_read(gpu, buf, 0, test_dst, buf_size));
    REQUIRE_MEMEQ(test_src, test_dst, buf_size);
    pl_buf_destroy(gpu, &buf);

    printf("- test buffer empty creation, update and readback\n");
    memset(test_dst, 0, buf_size);
    buf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .host_writable = true,
        .host_readable = true,
    ));

    REQUIRE(buf);
    pl_buf_write(gpu, buf, 0, test_src, buf_size);
    REQUIRE(pl_buf_read(gpu, buf, 0, test_dst, buf_size));
    REQUIRE_MEMEQ(test_src, test_dst, buf_size);
    pl_buf_destroy(gpu, &buf);

    printf("- test buffer-buffer copy and readback\n");
    memset(test_dst, 0, buf_size);
    buf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .initial_data = test_src,
    ));

    tbuf = pl_buf_create(gpu, pl_buf_params(
        .size = buf_size,
        .host_readable = true,
    ));

    REQUIRE(buf && tbuf);
    pl_buf_copy(gpu, tbuf, 0, buf, 0, buf_size);
    REQUIRE(pl_buf_read(gpu, tbuf, 0, test_dst, buf_size));
    REQUIRE_MEMEQ(test_src, test_dst, buf_size);
    pl_buf_destroy(gpu, &buf);
    pl_buf_destroy(gpu, &tbuf);

    if (buf_size <= gpu->limits.max_mapped_size) {
        printf("- test host mapped buffer readback\n");
        buf = pl_buf_create(gpu, pl_buf_params(
            .size = buf_size,
            .host_mapped = true,
            .initial_data = test_src,
        ));

        REQUIRE(buf);
        REQUIRE(!pl_buf_poll(gpu, buf, 0));
        REQUIRE_MEMEQ(test_src, buf->data, buf_size);
        pl_buf_destroy(gpu, &buf);
    }

    // `compute_queues` check is to exclude dummy GPUs here
    if (buf_size <= gpu->limits.max_ssbo_size && gpu->limits.compute_queues)
    {
        printf("- test endian swapping\n");
        buf = pl_buf_create(gpu, pl_buf_params(
            .size = buf_size,
            .storable = true,
            .initial_data = test_src,
        ));

        tbuf = pl_buf_create(gpu, pl_buf_params(
            .size = buf_size,
            .storable = true,
            .host_readable = true,
        ));

        REQUIRE(buf && tbuf);
        REQUIRE(pl_buf_copy_swap(gpu, &(struct pl_buf_copy_swap_params) {
            .src = buf,
            .dst = tbuf,
            .size = buf_size,
            .wordsize = 2,
        }));
        REQUIRE(pl_buf_read(gpu, tbuf, 0, test_dst, buf_size));
        for (int i = 0; i < buf_size / 2; i++) {
            REQUIRE_CMP(test_src[2 * i + 0], ==, test_dst[2 * i + 1], PRIu8);
            REQUIRE_CMP(test_src[2 * i + 1], ==, test_dst[2 * i + 0], PRIu8);
        }
        // test endian swap in-place
        REQUIRE(pl_buf_copy_swap(gpu, &(struct pl_buf_copy_swap_params) {
            .src = tbuf,
            .dst = tbuf,
            .size = buf_size,
            .wordsize = 4,
        }));
        REQUIRE(pl_buf_read(gpu, tbuf, 0, test_dst, buf_size));
        for (int i = 0; i < buf_size / 4; i++) {
            REQUIRE_CMP(test_src[4 * i + 0], ==, test_dst[4 * i + 2], PRIu8);
            REQUIRE_CMP(test_src[4 * i + 1], ==, test_dst[4 * i + 3], PRIu8);
            REQUIRE_CMP(test_src[4 * i + 2], ==, test_dst[4 * i + 0], PRIu8);
            REQUIRE_CMP(test_src[4 * i + 3], ==, test_dst[4 * i + 1], PRIu8);
        }
        pl_buf_destroy(gpu, &buf);
        pl_buf_destroy(gpu, &tbuf);
    }

    free(test_src);
}

static void test_cb(void *priv)
{
    bool *flag = priv;
    *flag = true;
}

static void pl_test_roundtrip(pl_gpu gpu, pl_tex tex[2],
                              uint8_t *src, uint8_t *dst)
{
    if (!tex[0] || !tex[1]) {
        printf("failed creating test textures... skipping this test\n");
        return;
    }

    int texels = tex[0]->params.w;
    texels *= tex[0]->params.h ? tex[0]->params.h : 1;
    texels *= tex[0]->params.d ? tex[0]->params.d : 1;

    pl_fmt fmt = tex[0]->params.format;
    size_t bytes = texels * fmt->texel_size;
    memset(src, 0, bytes);
    memset(dst, 0, bytes);

    for (int i = 0; i < texels; i++) {
        uint8_t *data = &src[i * fmt->texel_size];
        if (fmt->type == PL_FMT_FLOAT) {
            for (int n = 0; n < fmt->num_components; n++) {
                switch (fmt->component_depth[n]) {
                case 16: *(uint16_t *) data = RANDOM_F16; data += 2; break;
                case 32: *(float *)    data = RANDOM_F32; data += 4; break;
                case 64: *(double *)   data = RANDOM_F64; data += 8; break;
                }
            }
        } else {
            for (int n = 0; n < fmt->texel_size; n++)
                data[n] = RANDOM_U8;
        }
    }

    pl_timer ul, dl;
    ul = pl_timer_create(gpu);
    dl = pl_timer_create(gpu);

    bool ran_ul = false, ran_dl = false;

    REQUIRE(pl_tex_upload(gpu, &(struct pl_tex_transfer_params){
        .tex = tex[0],
        .ptr = src,
        .timer = ul,
        .callback = gpu->limits.callbacks ? test_cb : NULL,
        .priv = &ran_ul,
    }));

    // Test blitting, if possible for this format
    pl_tex dst_tex = tex[0];
    if (tex[0]->params.blit_src && tex[1]->params.blit_dst) {
        pl_tex_clear_ex(gpu, tex[1], (union pl_clear_color){0}); // for testing
        pl_tex_blit(gpu, &(struct pl_tex_blit_params) {
            .src = tex[0],
            .dst = tex[1],
        });
        dst_tex = tex[1];
    }

    REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params){
        .tex = dst_tex,
        .ptr = dst,
        .timer = dl,
        .callback = gpu->limits.callbacks ? test_cb : NULL,
        .priv = &ran_dl,
    }));

    pl_gpu_finish(gpu);
    if (gpu->limits.callbacks)
        REQUIRE(ran_ul && ran_dl);

    if (fmt->emulated && fmt->type == PL_FMT_FLOAT) {
        // TODO: can't memcmp here because bits might be lost due to the
        // emulated 16/32 bit upload paths, figure out a better way to
        // generate data and verify the roundtrip!
    } else {
        REQUIRE_MEMEQ(src, dst, bytes);
    }

    // Exercise timer queries
    (void) pl_timer_query(gpu, ul);
    (void) pl_timer_query(gpu, dl);

    pl_timer_destroy(gpu, &ul);
    pl_timer_destroy(gpu, &dl);
}

void pl_texture_tests(pl_gpu gpu)
{
    const size_t max_size = 16*16*16 * 4 *sizeof(double);
    uint8_t *test_src = malloc(max_size * 2);
    uint8_t *test_dst = test_src + max_size;
    printf("pl_texture_tests:\n");

    for (int f = 0; f < gpu->num_formats; f++) {
        pl_fmt fmt = gpu->formats[f];
        if (fmt->opaque || !(fmt->caps & PL_FMT_CAP_HOST_READABLE))
            continue;

        printf("- testing texture roundtrip for format %s\n", fmt->name);
        assert(fmt->texel_size <= 4 * sizeof(double));

        struct pl_tex_params ref_params = {
            .format        = fmt,
            .blit_src      = (fmt->caps & PL_FMT_CAP_BLITTABLE),
            .blit_dst      = (fmt->caps & PL_FMT_CAP_BLITTABLE),
            .host_writable = true,
            .host_readable = true,
            .debug_tag     = PL_DEBUG_TAG,
        };

        pl_tex tex[2];

        if (gpu->limits.max_tex_1d_dim >= 16) {
            struct pl_tex_params params = ref_params;
            params.w = 16;
            if (!gpu->limits.blittable_1d_3d)
                params.blit_src = params.blit_dst = false;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }

        if (gpu->limits.max_tex_2d_dim >= 16) {
            struct pl_tex_params params = ref_params;
            params.w = params.h = 16;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }

        if (gpu->limits.max_tex_3d_dim >= 16) {
            struct pl_tex_params params = ref_params;
            params.w = params.h = params.d = 16;
            if (!gpu->limits.blittable_1d_3d)
                params.blit_src = params.blit_dst = false;
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                tex[i] = pl_tex_create(gpu, &params);
            pl_test_roundtrip(gpu, tex, test_src, test_dst);
            for (int i = 0; i < PL_ARRAY_SIZE(tex); i++)
                pl_tex_destroy(gpu, &tex[i]);
        }
    }

    free(test_src);
}

static void pl_planar_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_named_fmt(gpu, "g8_b8_r8_420");
    if (!fmt)
        return;
    printf("pl_planar_tests:\n");
    REQUIRE_CMP(fmt->num_planes, ==, 3, "d");

    const int width = 64, height = 32;
    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w              = width,
        .h              = height,
        .format         = fmt,
        .blit_dst       = true,
        .host_readable  = true,
    ));
    if (!tex)
        return;
    for (int i = 0; i < fmt->num_planes; i++)
        REQUIRE(tex->planes[i]);

    pl_tex plane = tex->planes[1];
    uint8_t data[(width * height) >> 2];
    REQUIRE_CMP(plane->params.w * plane->params.h, ==, PL_ARRAY_SIZE(data), "d");

    pl_tex_clear(gpu, plane, (float[]){ (float) 0x80 / 0xFF, 0.0, 0.0, 1.0 });
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = plane,
        .ptr = data,
    )));

    uint8_t ref[PL_ARRAY_SIZE(data)];
    memset(ref, 0x80, sizeof(ref));
    REQUIRE_MEMEQ(data, ref, PL_ARRAY_SIZE(data));

    pl_tex_destroy(gpu, &tex);
}

static void pl_shader_tests(pl_gpu gpu)
{
    if (gpu->glsl.version < 410)
        return;
    printf("pl_shader_tests:\n");

    const char *vert_shader =
        "#version 410                               \n"
        "layout(location=0) in vec2 vertex_pos;     \n"
        "layout(location=1) in vec3 vertex_color;   \n"
        "layout(location=0) out vec3 frag_color;    \n"
        "void main() {                              \n"
        "    gl_Position = vec4(vertex_pos, 0, 1);  \n"
        "    frag_color = vertex_color;             \n"
        "}";

    const char *frag_shader =
        "#version 410                               \n"
        "layout(location=0) in vec3 frag_color;     \n"
        "layout(location=0) out vec4 out_color;     \n"
        "void main() {                              \n"
        "    out_color = vec4(frag_color, 1.0);     \n"
        "}";

    pl_fmt fbo_fmt;
    enum pl_fmt_caps caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_BLITTABLE |
                            PL_FMT_CAP_LINEAR;

    fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 32, caps);
    if (!fbo_fmt)
        return;

#define FBO_W 16
#define FBO_H 16

    pl_tex fbo;
    fbo = pl_tex_create(gpu, &(struct pl_tex_params) {
        .format         = fbo_fmt,
        .w              = FBO_W,
        .h              = FBO_H,
        .renderable     = true,
        .storable       = !!(fbo_fmt->caps & PL_FMT_CAP_STORABLE),
        .host_readable  = true,
        .blit_dst       = true,
    });
    REQUIRE(fbo);

    pl_tex_clear_ex(gpu, fbo, (union pl_clear_color){0});

    pl_fmt vert_fmt;
    vert_fmt = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 3);
    REQUIRE(vert_fmt);

    static const struct vertex { float pos[2]; float color[3]; } vertices[] = {
        {{-1.0, -1.0}, {0, 0, 0}},
        {{ 1.0, -1.0}, {1, 0, 0}},
        {{-1.0,  1.0}, {0, 1, 0}},
        {{ 1.0,  1.0}, {1, 1, 0}},
    };

    pl_pass pass;
    pass = pl_pass_create(gpu, &(struct pl_pass_params) {
        .type           = PL_PASS_RASTER,
        .target_format  = fbo_fmt,
        .vertex_shader  = vert_shader,
        .glsl_shader    = frag_shader,

        .vertex_type    = PL_PRIM_TRIANGLE_STRIP,
        .vertex_stride  = sizeof(struct vertex),
        .num_vertex_attribs = 2,
        .vertex_attribs = (struct pl_vertex_attrib[]) {{
            .name     = "vertex_pos",
            .fmt      = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            .location = 0,
            .offset   = offsetof(struct vertex, pos),
        }, {
            .name     = "vertex_color",
            .fmt      = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 3),
            .location = 1,
            .offset   = offsetof(struct vertex, color),
        }},
    });
    REQUIRE(pass);
    if (pass->params.cached_program || pass->params.cached_program_len) {
        // Ensure both are set if either one is set
        REQUIRE(pass->params.cached_program);
        REQUIRE(pass->params.cached_program_len);
    }

    pl_timer timer = pl_timer_create(gpu);
    pl_pass_run(gpu, &(struct pl_pass_run_params) {
        .pass           = pass,
        .target         = fbo,
        .vertex_count   = PL_ARRAY_SIZE(vertices),
        .vertex_data    = vertices,
        .timer          = timer,
    });

    // Wait until this pass is complete and report the timer result
    pl_gpu_finish(gpu);
    printf("timer query result: %"PRIu64"\n", pl_timer_query(gpu, timer));
    pl_timer_destroy(gpu, &timer);

    static float test_data[FBO_H * FBO_W * 4] = {0};

    // Test against the known pattern of `src`, only useful for roundtrip tests
#define TEST_FBO_PATTERN(eps, fmt, ...)                                     \
    do {                                                                    \
        printf("- testing pattern of " fmt "\n", __VA_ARGS__);                \
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {     \
            .tex = fbo,                                                     \
            .ptr = test_data,                                               \
        }));                                                                \
                                                                            \
        for (int y = 0; y < FBO_H; y++) {                                   \
            for (int x = 0; x < FBO_W; x++) {                               \
                float *color = &test_data[(y * FBO_W + x) * 4];             \
                REQUIRE_FEQ(color[0], (x + 0.5) / FBO_W, eps);              \
                REQUIRE_FEQ(color[1], (y + 0.5) / FBO_H, eps);              \
                REQUIRE_FEQ(color[2], 0.0, eps);                            \
                REQUIRE_FEQ(color[3], 1.0, eps);                            \
            }                                                               \
        }                                                                   \
    } while (0)

    TEST_FBO_PATTERN(1e-6, "%s", "initial rendering");

    if (sizeof(vertices) <= gpu->limits.max_vbo_size) {
        // Test the use of an explicit vertex buffer
        pl_buf vert = pl_buf_create(gpu, &(struct pl_buf_params) {
            .size = sizeof(vertices),
            .initial_data = vertices,
            .drawable = true,
        });

        REQUIRE(vert);
        pl_pass_run(gpu, &(struct pl_pass_run_params) {
            .pass           = pass,
            .target         = fbo,
            .vertex_count   = sizeof(vertices) / sizeof(struct vertex),
            .vertex_buf     = vert,
            .buf_offset     = 0,
        });

        pl_buf_destroy(gpu, &vert);
        TEST_FBO_PATTERN(1e-6, "%s", "using vertex buffer");
    }

    // Test the use of index buffers
    static const uint16_t indices[] = { 3, 2, 1, 0 };
    pl_pass_run(gpu, &(struct pl_pass_run_params) {
        .pass           = pass,
        .target         = fbo,
        .vertex_count   = PL_ARRAY_SIZE(indices),
        .vertex_data    = vertices,
        .index_data     = indices,
    });

    pl_pass_destroy(gpu, &pass);
    TEST_FBO_PATTERN(1e-6, "%s", "using indexed rendering");

    // Test the use of pl_dispatch
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .body       = "color = vec4(col, 1.0);",
        .input      = PL_SHADER_SIG_NONE,
        .output     = PL_SHADER_SIG_COLOR,
    }));

    REQUIRE(pl_dispatch_vertex(dp, &(struct pl_dispatch_vertex_params) {
        .shader         = &sh,
        .target         = fbo,
        .vertex_stride  = sizeof(struct vertex),
        .vertex_position_idx = 0,
        .num_vertex_attribs = 2,
        .vertex_attribs = (struct pl_vertex_attrib[]) {{
            .name   = "pos",
            .fmt    = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 2),
            .offset = offsetof(struct vertex, pos),
        }, {
            .name   = "col",
            .fmt    = pl_find_vertex_fmt(gpu, PL_FMT_FLOAT, 3),
            .offset = offsetof(struct vertex, color),
        }},

        .vertex_type    = PL_PRIM_TRIANGLE_STRIP,
        .vertex_coords  = PL_COORDS_NORMALIZED,
        .vertex_count   = PL_ARRAY_SIZE(vertices),
        .vertex_data    = vertices,
    }));

    TEST_FBO_PATTERN(1e-6, "%s", "using custom vertices");

    static float src_data[FBO_H * FBO_W * 4] = {0};
    memcpy(src_data, test_data, sizeof(src_data));

    pl_tex src;
    src = pl_tex_create(gpu, &(struct pl_tex_params) {
        .format         = fbo_fmt,
        .w              = FBO_W,
        .h              = FBO_H,
        .storable       = fbo->params.storable,
        .sampleable     = true,
        .initial_data   = src_data,
    });

    if (fbo->params.storable) {
        // Test 1x1 blit, to make sure the scaling code runs
        REQUIRE(pl_tex_blit_compute(gpu, &(struct pl_tex_blit_params) {
            .src = src,
            .dst = fbo,
            .src_rc = {0, 0, 0, 1, 1, 1},
            .dst_rc = {0, 0, 0, FBO_W, FBO_H, 1},
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        }));

        // Test non-resizing blit, which uses the efficient imageLoad path
        REQUIRE(pl_tex_blit_compute(gpu, &(struct pl_tex_blit_params) {
            .src = src,
            .dst = fbo,
            .src_rc = {0, 0, 0, FBO_W, FBO_H, 1},
            .dst_rc = {0, 0, 0, FBO_W, FBO_H, 1},
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        }));

        TEST_FBO_PATTERN(1e-6, "%s", "pl_tex_blit_compute");
    }

    // Test encoding/decoding of all gamma functions, color spaces, etc.
    for (enum pl_color_transfer trc = 0; trc < PL_COLOR_TRC_COUNT; trc++) {
        struct pl_color_space test_csp = {
            .transfer = trc,
            .hdr.min_luma = PL_COLOR_HDR_BLACK,
        };
        sh = pl_dispatch_begin(dp);
        pl_shader_sample_nearest(sh, pl_sample_src( .tex = src ));
        if (trc == PL_COLOR_TRC_LINEAR_HDR)
            GLSL("color.rgb = color.rgb * 100.0 - 25.0; \n");
        pl_shader_delinearize(sh, &test_csp);
        pl_shader_linearize(sh, &test_csp);
        if (trc == PL_COLOR_TRC_LINEAR_HDR)
            GLSL("color.rgb = (color.rgb + 25.0) / 100.0; \n");
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &sh,
            .target = fbo,
        )));

        float epsilon = pl_color_transfer_is_normalized_linear(trc) ? 1e-6 :
                        pl_color_transfer_is_hdr(trc) ? 1e-4 : 1e-6;
        TEST_FBO_PATTERN(epsilon, "transfer function: %s", pl_color_transfer_name(trc));
    }

    // Test scRGB. Mostly to make sure we output correct scale and out-of-gamut values.
    // BT.2020 LINEAR to scRGB with Display P3 display primaries.
    {
        static const float bt2020_in[][4] = {
            { 1.0f, 1.0f, 1.0f, 1.0f }, // white
            { 1.0f, 0.0f, 0.0f, 1.0f }, // red
            { 0.0f, 1.0f, 0.0f, 1.0f }, // green
            { 0.0f, 0.0f, 1.0f, 1.0f }, // blue
            { 0.5f, 0.5f, 0.5f, 1.0f }, // grays
            { 0.01f, 0.01f, 0.01f, 1.0f },
            { 0.0f, 0.0f, 0.0f, 1.0f },
        };

        static const float clip_ref[][3] = {
            {  2.5375f,   2.5375f,   2.5375f  },
            {  4.2135f,  -0.3160f,  -0.0461f  },
            { -1.4911f,   2.8747f,  -0.2552f  },
            { -0.1849f,  -0.0212f,   2.8388f  },
        };

        static const float sat_ref[][3] = {
            {  2.5375f,   2.5375f,   2.5375f  },
            {  3.1083f,  -0.1067f,  -0.0464f  },
            { -0.5708f,   2.6442f,  -0.1995f  },
            {  0.0000f,   0.0000f,   2.7834f  },
        };

        const int N = PL_ARRAY_SIZE(bt2020_in);

        pl_tex bt2020_tex = pl_tex_create(gpu, &(struct pl_tex_params) {
            .format       = fbo_fmt,
            .w            = N,
            .h            = 1,
            .sampleable   = true,
            .initial_data = bt2020_in,
        });
        REQUIRE(bt2020_tex);

        pl_tex ref_fbo = pl_tex_create(gpu, &(struct pl_tex_params) {
            .format        = fbo_fmt,
            .w             = N,
            .h             = 1,
            .renderable    = true,
            .storable      = !!(fbo_fmt->caps & PL_FMT_CAP_STORABLE),
            .host_readable = true,
        });
        REQUIRE(ref_fbo);

        for (int g = 0; g < pl_num_gamut_map_functions; g++) {
            const float (*ref)[3] = NULL;
            if (pl_gamut_map_functions[g] == &pl_gamut_map_clip)
                ref = clip_ref;
            else if (pl_gamut_map_functions[g] == &pl_gamut_map_saturation)
                ref = sat_ref;

            pl_shader_obj gmap_state = NULL;
            sh = pl_dispatch_begin(dp);
            pl_shader_sample_nearest(sh, pl_sample_src( .tex = bt2020_tex ));
            pl_shader_color_map(sh, &(struct pl_color_map_params) {
                                    .gamut_mapping = pl_gamut_map_functions[g],
                                },
                                (struct pl_color_space) {
                                    .primaries = PL_COLOR_PRIM_BT_2020,
                                    .transfer  = PL_COLOR_TRC_LINEAR,
                                },
                                (struct pl_color_space) {
                                    .primaries = PL_COLOR_PRIM_BT_709,
                                    .transfer  = PL_COLOR_TRC_SCRGB,
                                    .hdr.prim  = *pl_raw_primaries_get(PL_COLOR_PRIM_DISPLAY_P3),
                                }, &gmap_state, false);
            REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
                .shader = &sh,
                .target = ref_fbo,
            )));
            pl_shader_obj_destroy(&gmap_state);

            static float p3_out[PL_ARRAY_SIZE(bt2020_in) * 4];
            REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
                .tex = ref_fbo,
                .ptr = p3_out,
            }));

            for (int i = 0; i < N; i++) {
                const float *c = &p3_out[i * 4], *in = bt2020_in[i];
                if (ref && i < PL_ARRAY_SIZE(clip_ref)) {
                    REQUIRE_FEQ(c[0], ref[i][0], 1e-4f);
                    REQUIRE_FEQ(c[1], ref[i][1], 1e-4f);
                    REQUIRE_FEQ(c[2], ref[i][2], 1e-4f);
                }

                // Grays stay neutral and scale like white in every mapping
                if (in[0] == in[1] && in[1] == in[2]) {
                    REQUIRE_FEQ(c[0], c[1], 1e-4f);
                    REQUIRE_FEQ(c[1], c[2], 1e-4f);
                    REQUIRE_FEQ(c[1], in[1] * p3_out[1], 2e-3f);
                }
            }
        }

        pl_tex_destroy(gpu, &bt2020_tex);
        pl_tex_destroy(gpu, &ref_fbo);
    }

    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        if (sys == PL_COLOR_SYSTEM_DOLBYVISION)
            continue; // requires metadata
        sh = pl_dispatch_begin(dp);
        pl_shader_sample_nearest(sh, pl_sample_src( .tex = src ));
        pl_shader_encode_color(sh, &(struct pl_color_repr) { .sys = sys });
        pl_shader_decode_color_ex(sh, pl_color_decode_args(
            .repr = &(struct pl_color_repr) { .sys = sys },
        ));
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));

        float epsilon;
        switch (sys) {
        case PL_COLOR_SYSTEM_BT_2020_C:
        case PL_COLOR_SYSTEM_XYZ:
            epsilon = 1e-5;
            break;

        case PL_COLOR_SYSTEM_BT_2100_HLG:
            // R and B near black come out of the square root segment of HLG,
            // which turns the float rounding of the decode into up to 5e-4
            epsilon = 1e-3;
            break;

        case PL_COLOR_SYSTEM_BT_2100_PQ:
            // Float evaluation of the ST 2084 curves loses up to 0.1 of signal
            // near black, where R and B are small differences of large L, M
            // and S and the curve is steepest
            continue;

        default: epsilon = 1e-6; break;
        }

        TEST_FBO_PATTERN(epsilon, "color system: %s", pl_color_system_name(sys));
    }

    // XYZ roundtrip through explicit (non-default) primaries
    sh = pl_dispatch_begin(dp);
    pl_shader_sample_nearest(sh, pl_sample_src( .tex = src ));
    pl_shader_encode_color_ex(sh, pl_color_encode_args(
        .repr       = &(struct pl_color_repr) { .sys = PL_COLOR_SYSTEM_XYZ },
        .primaries  = PL_COLOR_PRIM_BT_2020,
    ));
    pl_shader_decode_color_ex(sh, pl_color_decode_args(
        .repr       = &(struct pl_color_repr) { .sys = PL_COLOR_SYSTEM_XYZ },
        .primaries  = PL_COLOR_PRIM_BT_2020,
    ));
    REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
    }));
    TEST_FBO_PATTERN(1e-5, "%s", "color system: XYZ (BT.2020 primaries)");

    // Repeat this a few times to test the caching
    pl_cache cache = pl_cache_create(pl_cache_params( .log = gpu->log ));
    pl_gpu_set_cache(gpu, cache);
    for (int i = 0; i < 10; i++) {
        if (i == 5) {
            printf("Recreating pl_dispatch to test the caching\n");
            size_t size = pl_dispatch_save(dp, NULL);
            REQUIRE(size);
            uint8_t *cache_data = malloc(size);
            REQUIRE(cache_data);
            REQUIRE_CMP(pl_dispatch_save(dp, cache_data), ==, size, "zu");

            pl_dispatch_destroy(&dp);
            dp = pl_dispatch_create(gpu->log, gpu);
            pl_dispatch_load(dp, cache_data);

            // Test to make sure the pass regenerates the same cache
            uint64_t hash = pl_str_hash((pl_str) { cache_data, size });
            REQUIRE_CMP(pl_dispatch_save(dp, NULL), ==, size, "zu");
            REQUIRE_CMP(pl_dispatch_save(dp, cache_data), ==, size, "zu");
            REQUIRE_CMP(pl_str_hash((pl_str) { cache_data, size }), ==, hash, PRIu64);
            free(cache_data);
        }

        sh = pl_dispatch_begin(dp);

        // For testing, force the use of CS if possible
        if (gpu->glsl.compute) {
            sh->type = SH_COMPUTE;
            sh->group_size[0] = 8;
            sh->group_size[1] = 8;
        }

        pl_shader_deband(sh, pl_sample_src( .tex = src ), pl_deband_params(
            .iterations     = 0,
            .grain          = 0.0,
        ));

        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
        TEST_FBO_PATTERN(1e-6, "deband iter %d", i);
    }

    pl_gpu_set_cache(gpu, NULL);
    pl_cache_destroy(&cache);

    // Test peak detection and readback if possible
    sh = pl_dispatch_begin(dp);
    pl_shader_sample_nearest(sh, pl_sample_src( .tex = src ));

    pl_shader_obj peak_state = NULL;
    struct pl_color_space csp_gamma22 = { .transfer = PL_COLOR_TRC_GAMMA22 };
    struct pl_peak_detect_params peak_params = { .minimum_peak = 0.01 };
    if (pl_shader_detect_peak(sh, csp_gamma22, &peak_state, &peak_params)) {
        REQUIRE(pl_dispatch_compute(dp, &(struct pl_dispatch_compute_params) {
            .shader = &sh,
            .width = fbo->params.w,
            .height = fbo->params.h,
        }));

        struct pl_hdr_metadata hdr;
        REQUIRE(pl_get_detected_hdr_metadata(peak_state, &hdr));

        float real_peak = 0, real_avg = 0;
        for (int y = 0; y < FBO_H; y++) {
            for (int x = 0; x < FBO_W; x++) {
                float *color = &src_data[(y * FBO_W + x) * 4];
                float luma = 0.212639f * powf(color[0], 2.2f) +
                             0.715169f * powf(color[1], 2.2f) +
                             0.072192f * powf(color[2], 2.2f);
                luma = pl_hdr_rescale(PL_HDR_NORM, PL_HDR_PQ, luma);
                real_peak = PL_MAX(real_peak, luma);
                real_avg += luma;
            }
        }
        real_avg = real_avg / (FBO_W * FBO_H);
        REQUIRE_FEQ(hdr.max_pq_y, real_peak, 1e-4);
        REQUIRE_FEQ(hdr.avg_pq_y, real_avg,  1e-3);
    }

    pl_dispatch_abort(dp, &sh);
    pl_shader_obj_destroy(&peak_state);

    // Test film grain synthesis
    pl_shader_obj grain = NULL;
    struct pl_film_grain_params grain_params = {
        .tex = src,
        .components = 3,
        .component_mapping = { 0, 1, 2},
        .repr = &(struct pl_color_repr) {
            .sys = PL_COLOR_SYSTEM_BT_709,
            .levels = PL_COLOR_LEVELS_LIMITED,
            .bits = { .color_depth = 10, .sample_depth = 10 },
        },
    };

    for (int i = 0; i < 2; i++) {
        grain_params.data.type = PL_FILM_GRAIN_AV1;
        grain_params.data.params.av1 = av1_grain_data;
        grain_params.data.params.av1.overlap = !!i;
        grain_params.data.seed = rand();

        sh = pl_dispatch_begin(dp);
        pl_shader_film_grain(sh, &grain, &grain_params);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
    }

    if (gpu->glsl.compute) {
        grain_params.data.type = PL_FILM_GRAIN_H274;
        grain_params.data.params.h274 = h274_grain_data;
        grain_params.data.seed = rand();

        sh = pl_dispatch_begin(dp);
        pl_shader_film_grain(sh, &grain, &grain_params);
        REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
            .shader = &sh,
            .target = fbo,
        }));
    }
    pl_shader_obj_destroy(&grain);

    // Test custom shaders
    struct pl_custom_shader custom = {
        .prelude =
            "#extension GL_ARB_texture_gather : enable  \n"
            "#define INVERT invert                      \n",

        .header =
            "vec3 invert(vec3 color)            \n"
            "{                                  \n"
            "    return vec3(1.0) - color;      \n"
            "}                                  \n",

        .body =
            "color = vec4(gl_FragCoord.xy, 0.0, 1.0);   \n"
            "color.rgb = INVERT(color.rgb) + offset;    \n",

        .input = PL_SHADER_SIG_NONE,
        .output = PL_SHADER_SIG_COLOR,

        .num_variables = 1,
        .variables = &(struct pl_shader_var) {
            .var = pl_var_float("offset"),
            .data = &(float) { 0.1 },
        },
    };

    sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_custom(sh, &custom));
    REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
    }));

    // Test dolbyvision
    struct pl_color_repr repr = {
        .sys = PL_COLOR_SYSTEM_DOLBYVISION,
        .dovi = &dovi_meta,
    };

    sh = pl_dispatch_begin(dp);
    pl_shader_sample_direct(sh, pl_sample_src( .tex = src ));
    pl_shader_decode_color_ex(sh, pl_color_decode_args( .repr = &repr ));
    REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
    }));

    // Test deinterlacing
    sh = pl_dispatch_begin(dp);
    pl_shader_deinterlace(sh, pl_deinterlace_source( .cur = pl_field_pair(src) ), NULL);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
        .shader = &sh,
        .target = fbo,
    )));

    // Test error diffusion
    if (fbo->params.storable) {
        for (int i = 0; i < pl_num_error_diffusion_kernels; i++) {
            const struct pl_error_diffusion_kernel *k = pl_error_diffusion_kernels[i];
            printf("- testing error diffusion kernel '%s'\n", k->name);
            sh = pl_dispatch_begin(dp);
            bool ok = pl_shader_error_diffusion(sh, pl_error_diffusion_params(
                .input_tex  = src,
                .output_tex = fbo,
                .new_depth  = 8,
                .kernel     = k,
            ));

            if (!ok) {
                fprintf(stderr, "kernel '%s' exceeds GPU limits, skipping...\n", k->name);
                continue;
            }

            REQUIRE(pl_dispatch_compute(dp, pl_dispatch_compute_params(
                .shader = &sh,
                .dispatch_size = {1, 1, 1},
            )));
        }
    }

    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &fbo);
}

static void pl_scaler_tests(pl_gpu gpu)
{
    pl_fmt src_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32, PL_FMT_CAP_LINEAR);
    pl_fmt fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 32, PL_FMT_CAP_RENDERABLE);
    if (!src_fmt || !fbo_fmt)
        return;
    printf("pl_scaler_tests:\n");

    float *fbo_data = NULL;
    pl_shader_obj lut = NULL;

    static float data_5x5[5][5] = {
        { 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0 },
        { 0, 0, 1, 0, 0 },
        { 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0 },
    };

    pl_tex dot5x5 = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w              = 5,
        .h              = 5,
        .format         = src_fmt,
        .sampleable     = true,
        .initial_data   = &data_5x5[0][0],
    });

    struct pl_tex_params fbo_params = {
        .w              = 100,
        .h              = 100,
        .format         = fbo_fmt,
        .renderable     = true,
        .storable       = fbo_fmt->caps & PL_FMT_CAP_STORABLE,
        .host_readable  = fbo_fmt->caps & PL_FMT_CAP_HOST_READABLE,
    };

    pl_tex fbo = pl_tex_create(gpu, &fbo_params);
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    if (!dot5x5 || !fbo || !dp)
        goto error;

    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_sample_polar(sh,
        pl_sample_src(
            .tex        = dot5x5,
            .new_w      = fbo->params.w,
            .new_h      = fbo->params.h,
        ),
        pl_sample_filter_params(
            .filter     = pl_filter_ewa_lanczos,
            .lut        = &lut,
            .no_compute = !fbo->params.storable,
        )
    ));
    REQUIRE(pl_dispatch_finish(dp, &(struct pl_dispatch_params) {
        .shader = &sh,
        .target = fbo,
    }));

    if (fbo->params.host_readable) {
        fbo_data = malloc(fbo->params.w * fbo->params.h * sizeof(float));
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
            .tex            = fbo,
            .ptr            = fbo_data,
        }));

#ifdef PRINT_OUTPUT
        int max = 255;
        printf("P2\n%d %d\n%d\n", fbo->params.w, fbo->params.h, max);
        for (int y = 0; y < fbo->params.h; y++) {
            for (int x = 0; x < fbo->params.w; x++) {
                float v = fbo_data[y * fbo->params.h + x];
                printf("%d ", (int) round(fmin(fmax(v, 0.0), 1.0) * max));
            }
            printf("\n");
        }
#endif
    }

error:
    free(fbo_data);
    pl_shader_obj_destroy(&lut);
    pl_dispatch_destroy(&dp);
    pl_tex_destroy(gpu, &dot5x5);
    pl_tex_destroy(gpu, &fbo);
}

static const char *user_shader_tests[] = {
    // Test hooking, saving and loading
    "// Example of a comment at the beginning                               \n"
    "                                                                       \n"
    "//!HOOK NATIVE                                                         \n"
    "//!DESC upscale image                                                  \n"
    "//!BIND HOOKED                                                         \n"
    "//!WIDTH HOOKED.w 10 *                                                 \n"
    "//!HEIGHT HOOKED.h 10 *                                                \n"
    "//!SAVE NATIVEBIG                                                      \n"
    "//!WHEN NATIVE.w 500 <                                                 \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return HOOKED_texOff(0);                                           \n"
    "}                                                                      \n"
    "                                                                       \n"
    "//!HOOK MAIN                                                           \n"
    "//!DESC downscale bigger image                                         \n"
    "//!WHEN NATIVE.w 500 <                                                 \n"
    "//!BIND NATIVEBIG                                                      \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return NATIVEBIG_texOff(0);                                        \n"
    "}                                                                      \n",

    // Test use of textures
    "//!HOOK MAIN                                                           \n"
    "//!DESC turn everything into colorful pixels                           \n"
    "//!BIND HOOKED                                                         \n"
    "//!BIND DISCO                                                          \n"
    "//!COMPONENTS 3                                                        \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return vec4(DISCO_tex(HOOKED_pos * 10.0).rgb, 1);                  \n"
    "}                                                                      \n"
    "                                                                       \n"
    "//!TEXTURE DISCO                                                       \n"
    "//!SIZE 3 3                                                            \n"
    "//!FORMAT rgba8                                                        \n"
    "//!FILTER NEAREST                                                      \n"
    "//!BORDER REPEAT                                                       \n"
    "ff0000ff00ff00ff0000ffff00ffffffff00ffffffff00ff4c4c4cff999999ffffffffff\n"

    // Test custom parameters
    "//!PARAM test                                                          \n"
    "//!DESC test parameter                                                 \n"
    "//!TYPE DYNAMIC float                                                  \n"
    "//!MINIMUM 0.0                                                         \n"
    "//!MAXIMUM 100.0                                                       \n"
    "1.0                                                                    \n"
    "                                                                       \n"
    "//!PARAM testconst                                                     \n"
    "//!TYPE CONSTANT uint                                                  \n"
    "//!MAXIMUM 16                                                          \n"
    "3                                                                      \n"
    "                                                                       \n"
    "//!PARAM testdefine                                                    \n"
    "//!TYPE DEFINE                                                         \n"
    "100                                                                    \n"
    "                                                                       \n"
    "//!PARAM testenum                                                      \n"
    "//!TYPE ENUM DEFINE                                                    \n"
    "FOO                                                                    \n"
    "BAR                                                                    \n"
    "                                                                       \n"
    "//!HOOK MAIN                                                           \n"
    "//!WHEN testconst 30 >                                                 \n"
    "#error should not be run                                               \n"
    "                                                                       \n"
    "//!HOOK MAIN                                                           \n"
    "//!WHEN testenum FOO =                                                 \n"
    "#if testenum == BAR                                                    \n"
    " #error bad                                                            \n"
    "#endif                                                                 \n"
    "vec4 hook() { return vec4(0.0); }                                      \n"
};

static const char *compute_shader_tests[] = {
    // Test use of storage/buffer resources
    "//!HOOK MAIN                                                           \n"
    "//!DESC attach some storage objects                                    \n"
    "//!BIND tex_storage                                                    \n"
    "//!BIND buf_uniform                                                    \n"
    "//!BIND buf_storage                                                    \n"
    "//!COMPONENTS 4                                                        \n"
    "                                                                       \n"
    "vec4 hook()                                                            \n"
    "{                                                                      \n"
    "    return vec4(foo, bar, bat);                                        \n"
    "}                                                                      \n"
    "                                                                       \n"
    "//!TEXTURE tex_storage                                                 \n"
    "//!SIZE 100 100                                                        \n"
    "//!FORMAT r32f                                                         \n"
    "//!STORAGE                                                             \n"
    "                                                                       \n"
    "//!BUFFER buf_uniform                                                  \n"
    "//!VAR float foo                                                       \n"
    "//!VAR float bar                                                       \n"
    "0000000000000000                                                       \n"
    "                                                                       \n"
    "//!BUFFER buf_storage                                                  \n"
    "//!VAR vec2 bat                                                        \n"
    "//!VAR int big[32];                                                    \n"
    "//!STORAGE                                                             \n",

};

static const char *test_luts[] = {

    "TITLE \"1D identity\"  \n"
    "LUT_1D_SIZE 2          \n"
    "0.0 0.0 0.0            \n"
    "1.0 1.0 1.0            \n",

    "TITLE \"3D identity\"  \n"
    "LUT_3D_SIZE 2          \n"
    "0.0 0.0 0.0            \n"
    "1.0 0.0 0.0            \n"
    "0.0 1.0 0.0            \n"
    "1.0 1.0 0.0            \n"
    "0.0 0.0 1.0            \n"
    "1.0 0.0 1.0            \n"
    "0.0 1.0 1.0            \n"
    "1.0 1.0 1.0            \n"

};

static bool frame_passthrough(pl_gpu gpu, pl_tex *tex,
                              const struct pl_source_frame *src, struct pl_frame *out_frame)
{
    const struct pl_frame *frame = src->frame_data;
    *out_frame = *frame;
    return true;
}

static enum pl_queue_status get_frame_ptr(struct pl_source_frame *out_frame,
                                          const struct pl_queue_params *qparams)
{
    const struct pl_source_frame **pframe = qparams->priv;
    if (!(*pframe)->frame_data)
        return PL_QUEUE_EOF;

    *out_frame = *(*pframe)++;
    return PL_QUEUE_OK;
}

static void render_info_cb(void *priv, const struct pl_render_info *info)
{
    REQUIRE(info->pass);
    REQUIRE(info->pass->shader->description);
}

static void peak_alpha_tests(pl_gpu gpu, enum pl_color_primaries primaries)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                             PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE |
                             PL_FMT_CAP_HOST_READABLE);
    if (!fmt || !gpu->limits.max_ssbo_size)
        return;
    float pixels[16 * 16 * 4], output[16 * 16 * 4];
    pl_tex texture = pl_tex_create(gpu, pl_tex_params(
        .w = 16, .h = 16, .format = fmt,
        .sampleable = true, .host_writable = true));
    pl_tex target = NULL;
    REQUIRE(texture);
    struct pl_color_space pq = pl_color_space_hdr10;
    pq.primaries = primaries;
    pq.hdr.prim = *pl_raw_primaries_get(primaries);
    pq.hdr.max_luma = 1000;
    pq.hdr.min_luma = 0.005f;
    const float codes[] = {0.7f, 0.5f, 0.3f};
    float linear[3];
    memcpy(linear, codes, sizeof(linear));
    pl_color_linearize(&pq, linear);
    const float alphas[] = {0, 0.5f, 1};
    for (int mode = 0; mode < 3; mode++) {
        int size = mode ? 8 : 16;
        REQUIRE(pl_tex_recreate(gpu, &target, pl_tex_params(
            .w = size, .h = size, .format = fmt,
            .renderable = true, .host_readable = true)));
        for (int a = 0; a < PL_ARRAY_SIZE(alphas); a++) {
            struct pl_hdr_metadata reference = {0};
            for (int variant = 0; variant < 6; variant++) {
                bool premultiplied = variant & 1;
                bool is_linear = variant >= 2;
                bool absolute = variant >= 4;
                for (int i = 0; i < 16 * 16; i++) {
                    for (int c = 0; c < 3; c++) {
                        float value = is_linear ? linear[c] : codes[c];
                        pixels[i * 4 + c] = premultiplied ? value * alphas[a] : value;
                    }
                    pixels[i * 4 + 3] = alphas[a];
                }
                REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
                    .tex = texture, .ptr = pixels)));
                struct pl_frame image = {
                    .num_planes = 1,
                    .planes[0] = {
                        .texture = texture, .components = 4,
                        .component_mapping = {0, 1, 2, 3},
                    },
                    .repr = pl_color_repr_rgb, .color = pq,
                };
                image.repr.alpha = premultiplied ? PL_ALPHA_PREMULTIPLIED
                                                 : PL_ALPHA_INDEPENDENT;
                if (is_linear)
                    image.color.transfer = absolute ? PL_COLOR_TRC_LINEAR_HDR
                                                     : PL_COLOR_TRC_LINEAR;
                struct pl_frame destination = image;
                destination.planes[0].texture = target;
                destination.repr = pl_color_repr_rgb;
                destination.color = pl_color_space_bt709;
                struct pl_render_params params = pl_render_default_params;
                params.disable_linear_scaling = mode == 2;
                params.peak_detect_params = pl_peak_detect_params(.allow_delayed = false);
                pl_renderer rr = pl_renderer_create(gpu->log, gpu);
                REQUIRE(rr);
                REQUIRE(pl_render_image(rr, &image, &destination, &params));
                REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
                    .tex = target, .ptr = output)));
                struct pl_hdr_metadata detected = {0};
                bool measured = pl_renderer_get_hdr_metadata(rr, &detected);
                printf("- peak alpha mode=%d alpha=%.1f variant=%d measured=%d\n",
                       mode, alphas[a], variant, measured);
                fflush(stdout);
                if (alphas[a] > 0) {
                    REQUIRE(measured);
                    if (!variant)
                        reference = detected;
                    REQUIRE(fabsf(detected.max_pq_y - reference.max_pq_y) < 0.001f);
                    REQUIRE(fabsf(detected.avg_pq_y - reference.avg_pq_y) < 0.001f);
                }
                for (int i = 0; i < size * size * 4; i++) {
                    REQUIRE(isfinite(output[i]));
                    if (!alphas[a] && i % 4 != 3)
                        REQUIRE(fabsf(output[i]) < 0.0001f);
                }
                REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
                pl_renderer_destroy(&rr);
            }
        }
    }
    pl_tex_destroy(gpu, &texture);
    pl_tex_destroy(gpu, &target);
}

static void pl_render_tests(pl_gpu gpu)
{
    pl_tex img_tex = NULL, fbo = NULL;
    pl_renderer rr = NULL;
    printf("pl_render_tests:\n");

    enum { width = 50, height = 50 };
    static float data[width][height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++)
            data[y][x] = RANDOM;
    }

    struct pl_plane img_plane = {0};
    struct pl_plane_data plane_data = {
        .type = PL_FMT_FLOAT,
        .width = width,
        .height = height,
        .component_size = { 8 * sizeof(float) },
        .component_map  = { 0 },
        .pixel_stride = sizeof(float),
        .pixels = data,
    };

    if (!pl_recreate_plane(gpu, NULL, &fbo, &plane_data))
        return;

    if (!pl_upload_plane(gpu, &img_plane, &img_tex, &plane_data))
        goto error;

    rr = pl_renderer_create(gpu->log, gpu);
    pl_tex_clear_ex(gpu, fbo, (union pl_clear_color){0});

    struct pl_frame image = {
        .num_planes     = 1,
        .planes         = { img_plane },
        .repr = {
            .sys        = PL_COLOR_SYSTEM_BT_709,
            .levels     = PL_COLOR_LEVELS_FULL,
        },
        .color          = pl_color_space_srgb,
    };

    struct pl_frame target = {
        .num_planes     = 1,
        .planes         = {{
            .texture            = fbo,
            .components         = 3,
            .component_mapping  = {0, 1, 2},
        }},
        .repr = {
            .sys        = PL_COLOR_SYSTEM_RGB,
            .levels     = PL_COLOR_LEVELS_FULL,
            .bits.color_depth = 32,
        },
        .color          = pl_color_space_srgb,
    };

    REQUIRE(pl_render_image(rr, &image, &target, NULL));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    // TODO: embed a reference texture and ensure it matches

    // Test a bunch of different params
#define TEST(SNAME, STYPE, DEFAULT, FIELD, LIMIT)                       \
    do {                                                                \
        for (int i = 0; i <= LIMIT; i++) {                              \
            printf("- testing `" #STYPE "." #FIELD " = %d`\n", i);        \
            struct pl_render_params params = pl_render_default_params;  \
            params.force_dither = true;                                 \
            struct STYPE tmp = DEFAULT;                                 \
            tmp.FIELD = i;                                              \
            params.SNAME = &tmp;                                        \
            REQUIRE(pl_render_image(rr, &image, &target, &params));     \
            pl_gpu_flush(gpu);                                          \
            REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE); \
        }                                                               \
    } while (0)

#define TEST_PARAMS(NAME, FIELD, LIMIT) \
    TEST(NAME##_params, pl_##NAME##_params, pl_##NAME##_default_params, FIELD, LIMIT)

    image.crop.x1 = width / 2.0;
    image.crop.y1 = height / 2.0;
    for (int i = 0; i < pl_num_scale_filters; i++) {
        struct pl_render_params params = pl_render_default_params;
        params.upscaler = pl_scale_filters[i].filter;
        printf("- testing `params.upscaler = /* %s */`\n", pl_scale_filters[i].name);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        pl_gpu_flush(gpu);
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }
    image.crop.x1 = image.crop.y1 = 0;

    target.crop.x1 = width / 2.0;
    target.crop.y1 = height / 2.0;
    for (int i = 0; i < pl_num_scale_filters; i++) {
        struct pl_render_params params = pl_render_default_params;
        params.downscaler = pl_scale_filters[i].filter;
        printf("- testing `params.downscaler = /* %s */`\n", pl_scale_filters[i].name);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        pl_gpu_flush(gpu);
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }
    target.crop.x1 = target.crop.y1 = 0;

    TEST_PARAMS(deband, iterations, 3);
    TEST_PARAMS(sigmoid, center, 1);
    TEST_PARAMS(color_map, intent, PL_INTENT_ABSOLUTE_COLORIMETRIC);
    TEST_PARAMS(dither, method, PL_DITHER_WHITE_NOISE);
    TEST_PARAMS(dither, temporal, true);
    TEST_PARAMS(distort, alpha_mode, PL_ALPHA_INDEPENDENT);
    TEST_PARAMS(distort, constrain, true);
    TEST_PARAMS(distort, bicubic, true);
    TEST(cone_params, pl_cone_params, pl_vision_deuteranomaly, strength, 0);

    // Test gamma-correct dithering
    target.repr.bits.color_depth = 2;
    TEST_PARAMS(dither, transfer, PL_COLOR_TRC_GAMMA22);
    target.repr.bits.color_depth = 32;

    // Test dithering with dynamic constants, which must leave the step count
    // a constant expression
    struct pl_render_params dyn_params = pl_render_default_params;
    dyn_params.force_dither = true;
    dyn_params.dynamic_constants = true;
    REQUIRE(pl_render_image(rr, &image, &target, &dyn_params));
    pl_gpu_flush(gpu);
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    // Test HDR tone mapping
    peak_alpha_tests(gpu, PL_COLOR_PRIM_BT_2020);
    peak_alpha_tests(gpu, PL_COLOR_PRIM_DISPLAY_P3);
    image.color = pl_color_space_hdr10;
    TEST_PARAMS(color_map, visualize_lut, true);
    if (gpu->limits.max_ssbo_size)
        TEST_PARAMS(peak_detect, allow_delayed, true);

    // A linear HDR frame can exceed nominal white, independently of its
    // mastering display. Peak detection must not infer a 203 nit upper bound.
    if (gpu->limits.max_ssbo_size) {
        const float targets[] = {203, 1000, 4000};
        struct pl_render_params peak_params = pl_render_default_params;
        peak_params.peak_detect_params = pl_peak_detect_params(.allow_delayed = false);
        struct pl_frame linear = image;
        linear.color.transfer = PL_COLOR_TRC_LINEAR;
        linear.color.hdr.max_luma = 1000;
        for (int i = 0; i < PL_ARRAY_SIZE(targets); i++) {
            struct pl_frame output = target;
            output.color.hdr.max_luma = targets[i];
            pl_renderer detector = pl_renderer_create(gpu->log, gpu);
            REQUIRE(detector);
            REQUIRE(pl_render_image(detector, &linear, &output, &peak_params));
            pl_gpu_finish(gpu);
            struct pl_hdr_metadata detected = {0};
            REQUIRE(pl_renderer_get_hdr_metadata(detector, &detected));
            REQUIRE(detected.max_pq_y > 0 && detected.avg_pq_y > 0);
            REQUIRE(pl_renderer_get_errors(detector).errors == PL_RENDER_ERR_NONE);
            pl_renderer_destroy(&detector);
        }
    }

    // Test inverse tone-mapping and pure BPC
    image.color.hdr.max_luma = 1000;
    target.color.hdr.max_luma = 4000;
    target.color.hdr.min_luma = 0.02;
    TEST_PARAMS(color_map, inverse_tone_mapping, true);

    image.color = pl_color_space_srgb;
    target.color = pl_color_space_srgb;

    // Test some misc stuff
    struct pl_render_params params = pl_render_default_params;
    params.color_adjustment = &(struct pl_color_adjustment) {
        .brightness = 0.1,
        .contrast = 0.9,
        .saturation = 1.5,
        .gamma = 0.8,
        .temperature = 0.3,
    };
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    params = pl_render_default_params;

    struct pl_frame inferred_image = image, inferred_target = target;
    pl_frames_infer(rr, &inferred_image, &inferred_target);
    REQUIRE(pl_render_image(rr, &inferred_image, &inferred_target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    // DV reconstruction defines the decoded color space, independently of
    // transport tags. Inferring it must preserve authored luminance metadata.
    const enum pl_color_primaries dv_prim[] = {
        PL_COLOR_PRIM_UNKNOWN, PL_COLOR_PRIM_BT_709, PL_COLOR_PRIM_DCI_P3,
    };
    const enum pl_color_transfer dv_trc[] = {
        PL_COLOR_TRC_UNKNOWN, PL_COLOR_TRC_SRGB, PL_COLOR_TRC_PQ,
    };
    for (int p = 0; p < PL_ARRAY_SIZE(dv_prim); p++) {
        for (int t = 0; t < PL_ARRAY_SIZE(dv_trc); t++) {
            struct pl_frame dv = image, output = target;
            dv.repr.sys = PL_COLOR_SYSTEM_DOLBYVISION;
            dv.color = pl_color_space_hdr10;
            dv.color.primaries = dv_prim[p];
            dv.color.transfer = dv_trc[t];
            dv.color.hdr.prim = *pl_raw_primaries_get(PL_COLOR_PRIM_BT_2020);
            dv.color.hdr.min_luma = 0.005f;
            dv.color.hdr.max_luma = 1000;
            dv.color.hdr.max_pq_y = 0.75f;
            dv.color.hdr.avg_pq_y = 0.5f;
            struct pl_hdr_metadata original = dv.color.hdr;
            pl_frames_infer(rr, &dv, &output);
            REQUIRE(dv.color.primaries == PL_COLOR_PRIM_BT_2020);
            REQUIRE(dv.color.transfer == PL_COLOR_TRC_PQ);
            REQUIRE(pl_hdr_metadata_equal(&dv.color.hdr, &original));
        }
    }

    // Test background blending and alpha transparency
    params.blend_against_tiles = true;
    params.corner_rounding = 0.25f;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    params = pl_render_default_params;

    // Test film grain synthesis
    image.film_grain.type = PL_FILM_GRAIN_AV1;
    image.film_grain.params.av1 = av1_grain_data;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    image.film_grain.type = PL_FILM_GRAIN_H274;
    image.film_grain.params.h274 = h274_grain_data;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    // H.274 film grain synthesis requires compute shaders
    if (gpu->glsl.compute) {
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    } else {
        const struct pl_render_errors rr_err = pl_renderer_get_errors(rr);
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_FILM_GRAIN);
        pl_renderer_reset_errors(rr, &rr_err);
    }
    image.film_grain = (struct pl_film_grain_data) {0};

    // Test mpv-style custom shaders
    for (int i = 0; i < PL_ARRAY_SIZE(user_shader_tests); i++) {
        printf("- testing user shader:\n\n%s\n", user_shader_tests[i]);
        const struct pl_hook *hook;
        hook = pl_mpv_user_shader_parse(gpu, user_shader_tests[i],
                                        strlen(user_shader_tests[i]));
        REQUIRE(hook);

        params.hooks = &hook;
        params.num_hooks = 1;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

        pl_mpv_user_shader_destroy(&hook);
    }

    if (gpu->glsl.compute && gpu->limits.max_ssbo_size) {
        for (int i = 0; i < PL_ARRAY_SIZE(compute_shader_tests); i++) {
            printf("- testing user shader:\n\n%s\n", compute_shader_tests[i]);
            const struct pl_hook *hook;
            hook = pl_mpv_user_shader_parse(gpu, compute_shader_tests[i],
                                            strlen(compute_shader_tests[i]));
            REQUIRE(hook);

            params.hooks = &hook;
            params.num_hooks = 1;
            REQUIRE(pl_render_image(rr, &image, &target, &params));
            REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

            pl_mpv_user_shader_destroy(&hook);
        }
    }
    params = pl_render_default_params;

    // Test custom LUTs
    for (int i = 0; i < PL_ARRAY_SIZE(test_luts); i++) {
        printf("- testing custom lut %d\n", i);
        struct pl_custom_lut *lut;
        lut = pl_lut_parse_cube(gpu->log, test_luts[i], strlen(test_luts[i]));
        REQUIRE(lut);

        bool has_3dlut = gpu->limits.max_tex_3d_dim && gpu->glsl.version > 100;
        if (lut->size[2] && !has_3dlut) {
            pl_lut_free(&lut);
            continue;
        }

        // Test all three at the same time to reduce the number of tests
        image.lut = target.lut = params.lut = lut;

        for (enum pl_lut_type t = PL_LUT_UNKNOWN; t <= PL_LUT_CONVERSION; t++) {
            printf("- testing LUT method %d\n", t);
            image.lut_type = target.lut_type = params.lut_type = t;
            REQUIRE(pl_render_image(rr, &image, &target, &params));
            REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        }

        image.lut = target.lut = params.lut = NULL;
        pl_lut_free(&lut);
    }

#ifdef PL_HAVE_LCMS

    // It doesn't fit without use of 3D textures on GLES2
    if (gpu->glsl.version > 100) {
        // Test ICC profiles
        image.profile = TEST_PROFILE(sRGB_v2_nano_icc);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        image.profile = (struct pl_icc_profile) {0};

        target.profile = TEST_PROFILE(sRGB_v2_nano_icc);
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        target.profile = (struct pl_icc_profile) {0};

        image.profile = TEST_PROFILE(sRGB_v2_nano_icc);
        target.profile = image.profile;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        image.profile = (struct pl_icc_profile) {0};
        target.profile = (struct pl_icc_profile) {0};
    }

#endif

    // Test overlays
    image.num_overlays = 1;
    image.overlays = &(struct pl_overlay) {
        .tex = img_plane.texture,
        .mode = PL_OVERLAY_NORMAL,
        .num_parts = 2,
        .parts = (struct pl_overlay_part[]) {{
            .src = {0, 0, 2, 2},
            .dst = {30, 100, 40, 200},
        }, {
            .src = {2, 2, 5, 5},
            .dst = {1000, -1, 3, 5},
        }},
    };
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    params.disable_fbos = true;
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    image.num_overlays = 0;
    params = pl_render_default_params;

    target.num_overlays = 1;
    target.overlays = &(struct pl_overlay) {
        .tex = img_plane.texture,
        .mode = PL_OVERLAY_MONOCHROME,
        .num_parts = 1,
        .parts = &(struct pl_overlay_part) {
            .src = {5, 5, 15, 15},
            .dst = {5, 5, 15, 15},
            .color = {1.0, 0.5, 0.0},
        },
    };
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    REQUIRE(pl_render_image(rr, NULL, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    target.num_overlays = 0;

    // Test rotation
    for (pl_rotation rot = 0; rot < PL_ROTATION_360; rot += PL_ROTATION_90) {
        image.rotation = rot;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }

    // Attempt frame mixing, using the mixer queue helper
    printf("- testing frame mixing\n");
    struct pl_render_params mix_params = {
        .frame_mixer = &pl_filter_mitchell_clamp,
        .info_callback = render_info_cb,
    };

    struct pl_queue_params qparams = {
        .radius = pl_frame_mix_radius(&mix_params),
        .vsync_duration = 1.0 / 60.0,
    };

    // Test large PTS jumps in frame mix
    struct pl_frame_mix mix = (struct pl_frame_mix) {
        .num_frames = 2,
        .frames = (const struct pl_frame *[]) { &image, &image },
        .signatures = (uint64_t[]) { 0xFFF1, 0xFFF2 },
        .timestamps = (double[]) { -100, 100 },
        .vsync_duration = 1.6,
    };
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test inferring frame mix
    inferred_target = target;
    pl_frames_infer_mix(rr, &mix, &inferred_target, &inferred_image);
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test empty frame mix
    mix = (struct pl_frame_mix) {0};
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test inferring empty frame mix
    inferred_target = target;
    pl_frames_infer_mix(rr, &mix, &inferred_target, &inferred_image);
    REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

    // Test mixer queue
#define NUM_MIX_FRAMES 20
    const float frame_duration = 1.0 / 24.0;
    struct pl_source_frame srcframes[NUM_MIX_FRAMES+1];
    srcframes[NUM_MIX_FRAMES] = (struct pl_source_frame) {0};
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        srcframes[i] = (struct pl_source_frame) {
            .pts = i * frame_duration,
            .duration = frame_duration,
            .map = frame_passthrough,
            .frame_data = &image,
        };
    }

    pl_queue queue = pl_queue_create(gpu);
    enum pl_queue_status ret;

    // Test pre-pushing all frames, with delayed EOF.
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        const struct pl_source_frame *src = &srcframes[i];
        if (i > 10) // test pushing in reverse order
            src = &srcframes[NUM_MIX_FRAMES + 10 - i];
        if (!pl_queue_push_block(queue, 1, src)) // mini-sleep
            pl_queue_push(queue, src); // push it anyway, for testing
    }

    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        if (ret == PL_QUEUE_MORE) {
            REQUIRE_CMP(qparams.pts, >, 0.0f, "f");
            pl_queue_push(queue, NULL); // push delayed EOF
            continue;
        }

        REQUIRE_CMP(ret, ==, PL_QUEUE_OK, "u");
        REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));

        // Simulate advancing vsync
        qparams.pts += qparams.vsync_duration;
    }

    // Test dynamically pulling all frames, with oversample mixer
    const struct pl_source_frame *frame_ptr = &srcframes[0];
    mix_params.frame_mixer = &pl_oversample_frame_mixer;

    qparams = (struct pl_queue_params) {
        .radius = pl_frame_mix_radius(&mix_params),
        .vsync_duration = qparams.vsync_duration,
        .get_frame = get_frame_ptr,
        .priv = &frame_ptr,
    };

    pl_queue_reset(queue);
    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        REQUIRE_CMP(ret, ==, PL_QUEUE_OK, "u");
        REQUIRE_CMP(mix.num_frames, <=, 2, "d");
        REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));
        qparams.pts += qparams.vsync_duration;
    }

    // Test large PTS jump
    pl_queue_reset(queue);
    REQUIRE(pl_queue_update(queue, &mix, &qparams) == PL_QUEUE_EOF);

    // Test that advancing without a mix still reports missing frames
    const struct pl_filter_config *mixers[] = {
        &pl_oversample_frame_mixer, &pl_filter_mitchell_clamp,
    };
    for (int m = 0; m < PL_ARRAY_SIZE(mixers); m++) {
        mix_params.frame_mixer = mixers[m];
        qparams = (struct pl_queue_params) {
            .radius = pl_frame_mix_radius(&mix_params),
            .vsync_duration = qparams.vsync_duration,
        };

        pl_queue_reset(queue);
        for (int i = 0; i < 4; i++)
            pl_queue_push(queue, &srcframes[i]);
        REQUIRE_CMP(pl_queue_update(queue, NULL, &qparams), ==, PL_QUEUE_OK, "u");
        qparams.pts = 3.5 * frame_duration;
        REQUIRE_CMP(pl_queue_update(queue, NULL, &qparams), ==, PL_QUEUE_MORE, "u");
    }

    // Test deinterlacing
    pl_queue_reset(queue);
    printf("- testing deinterlacing\n");
    for (int i = 0; i < NUM_MIX_FRAMES; i++) {
        struct pl_source_frame *src = &srcframes[i];
        if (i > 10)
            src = &srcframes[NUM_MIX_FRAMES + 10 - i];
        src->first_field = PL_FIELD_EVEN;
        pl_queue_push(queue, src);
    }
    pl_queue_push(queue, NULL);

    qparams.pts = 0;
    qparams.get_frame = NULL;
    while ((ret = pl_queue_update(queue, &mix, &qparams)) != PL_QUEUE_EOF) {
        REQUIRE_CMP(ret, ==, PL_QUEUE_OK, "u");
        REQUIRE(pl_render_image_mix(rr, &mix, &target, &mix_params));
        qparams.pts += qparams.vsync_duration;
    }

    pl_queue_destroy(&queue);

error:
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &img_tex);
    pl_tex_destroy(gpu, &fbo);
}

// Tests for the bit encoding logic, in both directions:
//  - encoding MSB-aligned output (e.g. P010) must produce the same value as
//    the LSB-aligned equivalent (e.g. yuv420p10), with zero padding bits
//  - decoding must survive a mismatch between the sample depth and the color
//    depth, for any way of padding the color samples into the texture
static void pl_render_bits_tests(pl_gpu gpu)
{
    pl_renderer rr = NULL;
    pl_dispatch dp = NULL;
    pl_tex src_tex = NULL, lsb_tex = NULL, msb_tex = NULL, fbo_tex = NULL;
    printf("pl_render_bits_tests:\n");

    // Need a true 16-bit UNORM render target we can read back bit-exactly
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 16, 16,
                             PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fmt || fmt->component_depth[0] != 16) {
        printf("- no suitable 16-bit UNORM format, skipping\n");
        return;
    }

    enum { width = 64, height = 64 };

    static uint16_t src_data[height][width][4];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint16_t v = (uint16_t) ((x + y * width) * 0xFFFFu / (width * height - 1));
            for (int c = 0; c < 4; c++)
                src_data[y][x][c] = v;
        }
    }

    struct pl_plane src_plane = {0};
    struct pl_plane_data src_pdata = {
        .type           = PL_FMT_UNORM,
        .width          = width,
        .height         = height,
        .component_size = { 16, 16, 16, 16 },
        .component_map  = { 0, 1, 2, 3 },
        .pixel_stride   = 4 * sizeof(uint16_t),
        .pixels         = src_data,
    };
    if (!pl_upload_plane(gpu, &src_plane, &src_tex, &src_pdata))
        goto error;

    lsb_tex = pl_tex_create(gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt,
        .renderable = true, .host_readable = true,
    ));
    msb_tex = pl_tex_create(gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt,
        .renderable = true, .host_readable = true,
    ));
    REQUIRE(lsb_tex);
    REQUIRE(msb_tex);

    rr = pl_renderer_create(gpu->log, gpu);
    REQUIRE(rr);

    struct pl_frame image = {
        .num_planes = 1,
        .planes     = { src_plane },
        .repr       = pl_color_repr_rgb,
        .color      = pl_color_space_srgb,
    };

    struct pl_frame target = {
        .num_planes = 1,
        .planes     = {{
            .components        = 3,
            .component_mapping = {0, 1, 2},
        }},
        .repr = {
            .sys    = PL_COLOR_SYSTEM_BT_2020_NC,
            .levels = PL_COLOR_LEVELS_LIMITED,
            .bits   = { .sample_depth = 16, .color_depth = 10 },
        },
        .color = pl_color_space_srgb,
    };

    static uint16_t lsb_out[height][width][4];
    static uint16_t msb_out[height][width][4];

    // Dithering puts both encodings on the same 10-bit grid, so they must match
    // exactly.
    for (int dither = 1; dither >= 0; dither--) {
        struct pl_render_params params = pl_render_default_params;
        if (!dither)
            params.dither_params = NULL;

        target.planes[0].texture = lsb_tex;
        target.repr.bits.bit_shift = 0;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

        target.planes[0].texture = msb_tex;
        target.repr.bits.bit_shift = 6;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = lsb_tex, .ptr = lsb_out)));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = msb_tex, .ptr = msb_out)));

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    int lsb = lsb_out[y][x][c];
                    int msb = msb_out[y][x][c];
                    REQUIRE_CMP(lsb, <=, 0x3FF, "d");    // fits in 10 bits
                    REQUIRE_CMP(msb & 0x3F, ==, 0, "d"); // padding bits are zero
                    if (dither)
                        REQUIRE_CMP(msb >> 6, ==, lsb, "d");
                }
            }
        }
    }

    // Dithering must land a flat gray on its exact code on average, and the
    // chroma on the neutral code, with the codes shifted up in the texture,
    // where their grid is not 2^depth - 1 steps. Error diffusion goes through
    // a 16-bit float FBO, which only holds 8-bit codes exactly enough.
    static const struct {
        int depth;
        const struct pl_error_diffusion_kernel *error_diffusion;
        double tolerance;
    } modes[] = {
        { 10, NULL, 0.02 },
        {  8, &pl_error_diffusion_sierra_lite, 0.2 },
    };

    const uint16_t gray = 0.6f * 0xFFFF;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 4; c++)
                src_data[y][x][c] = c < 3 ? gray : 0xFFFF;
        }
    }
    REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = src_tex, .ptr = src_data)));
    target.planes[0].texture = msb_tex;

    for (int m = 0; m < PL_ARRAY_SIZE(modes); m++) {
        const int depth = modes[m].depth, shift = 16 - depth;
        struct pl_render_params params = pl_render_default_params;
        params.error_diffusion = modes[m].error_diffusion;
        target.repr.bits.color_depth = depth;
        target.repr.bits.bit_shift = shift;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = msb_tex, .ptr = msb_out)));

        // Limited range luma and neutral chroma, at `depth` bits
        const double code[3] = {
            (16 + 219 * (gray / 65535.0)) * (1 << (depth - 8)),
            1 << (depth - 1),
            1 << (depth - 1),
        };
        for (int c = 0; c < 3; c++) {
            double error = 0.0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++)
                    error += (msb_out[y][x][c] >> shift) - code[c];
            }
            REQUIRE_CMP(fabs(error / (width * height)), <, modes[m].tolerance, "f");
        }
    }

    // The decode path has to survive a mismatch between the sample depth and
    // the color depth: neutral chroma must stay an exact grayscale, and the
    // nominal luma range must keep mapping onto exactly [0, 1], no matter how
    // the color samples are padded into the texture.
    pl_fmt f32 = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                             PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!f32 || !(fmt->caps & PL_FMT_CAP_SAMPLEABLE)) {
        printf("- no suitable f32 fmt, skipping decode tests\n");
        goto error;
    }

    dp = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(dp);

    // Luma test points, as a fraction of the nominal luma range
    static const float points[] = { 0.0, 0.25, 0.5, 0.75, 1.0 };
    enum { num_points = PL_ARRAY_SIZE(points) };

    pl_tex_destroy(gpu, &src_tex);
    src_tex = pl_tex_create(gpu, pl_tex_params(
        .w = num_points, .h = 1, .format = fmt,
        .sampleable = true, .host_writable = true,
    ));

    fbo_tex = pl_tex_create(gpu, pl_tex_params(
        .w = num_points, .h = 1, .format = f32,
        .renderable = true, .host_readable = true,
    ));
    if (!fbo_tex)
        goto error;

    static const struct { int color_depth, bit_shift; } encodings[] = {
        { 16, 0 }, // control: no padding at all
        { 12, 0 }, // 12-bit LSB-aligned
        { 12, 4 }, // 12-bit MSB-aligned (e.g. xyz12)
        { 10, 0 }, // 10-bit LSB-aligned
        {  8, 0 }, //  8-bit reference
    };

    for (int e = 0; e < PL_ARRAY_SIZE(encodings); e++) {
        const int col = encodings[e].color_depth;
        const int shift = encodings[e].bit_shift;

        for (int l = 0; l < 2; l++) {
            const enum pl_color_levels levels = l ? PL_COLOR_LEVELS_FULL
                                                  : PL_COLOR_LEVELS_LIMITED;
            const struct pl_color_repr repr0 = {
                .sys    = PL_COLOR_SYSTEM_BT_709,
                .levels = levels,
                .bits   = {
                    .sample_depth = 16,
                    .color_depth  = col,
                    .bit_shift    = shift,
                },
            };
            struct pl_color_repr repr = repr0;

            // Nominal luma range and neutral chroma point, at `col` bits
            const int ylo = levels == PL_COLOR_LEVELS_LIMITED ?  16 << (col - 8) : 0;
            const int yhi = levels == PL_COLOR_LEVELS_LIMITED ? 235 << (col - 8)
                                                              : (1 << col) - 1;
            const int neutral = 1 << (col - 1);

            uint16_t data[num_points][4];
            float expected[num_points];
            for (int i = 0; i < num_points; i++) {
                // Derive the expectation from the rounded code, not from the
                // fraction, so that the test pins the exact mapping
                const int luma = roundf(ylo + points[i] * (yhi - ylo));
                expected[i] = (float) (luma - ylo) / (yhi - ylo);
                data[i][0] = luma    << shift;
                data[i][1] = neutral << shift;
                data[i][2] = neutral << shift;
                data[i][3] = 0xFFFF;
            }

            REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
                .tex = src_tex,
                .ptr = data,
            )));

            pl_shader sh = pl_dispatch_begin(dp);
            pl_shader_sample_direct(sh, pl_sample_src( .tex = src_tex ));
            pl_shader_decode_color_ex(sh, pl_color_decode_args( .repr = &repr ));
            REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
                .shader = &sh,
                .target = fbo_tex,
            )));

            float out[num_points][4];
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
                .tex = fbo_tex,
                .ptr = out,
            )));

            printf("- %s range, color_depth=%d bit_shift=%d\n",
                   levels == PL_COLOR_LEVELS_FULL ? "full" : "limited",
                   col, shift);

            for (int i = 0; i < num_points; i++) {
                const float r = out[i][0], g = out[i][1], b = out[i][2];
                REQUIRE_FEQ(g, expected[i], 1e-6);
                REQUIRE_FEQ(r, g, 1e-6);
                REQUIRE_FEQ(r, b, 1e-6);
            }

            // The same through the renderer, which normalizes the planes
            // before the decode, and makes up the neutral chroma of a
            // luma-only image itself
            for (int luma_only = 0; luma_only < 2; luma_only++) {
                struct pl_frame src_frame = {
                    .num_planes = 1,
                    .planes = {{
                        .texture           = src_tex,
                        .components        = luma_only ? 1 : 3,
                        .component_mapping = {0, 1, 2},
                    }},
                    .repr  = repr0,
                    .color = pl_color_space_bt709,
                };
                struct pl_frame dst_frame = {
                    .num_planes = 1,
                    .planes = {{
                        .texture           = fbo_tex,
                        .components        = 3,
                        .component_mapping = {0, 1, 2},
                    }},
                    .repr  = pl_color_repr_rgb,
                    .color = pl_color_space_bt709,
                };
                struct pl_render_params params = pl_render_default_params;
                params.dither_params = NULL;
                REQUIRE(pl_render_image(rr, &src_frame, &dst_frame, &params));
                REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
                REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
                    .tex = fbo_tex,
                    .ptr = out,
                )));

                for (int i = 0; i < num_points; i++) {
                    const float r = out[i][0], g = out[i][1], b = out[i][2];
                    REQUIRE_FEQ(g, expected[i], 1e-6);
                    REQUIRE_FEQ(r, g, 1e-6);
                    REQUIRE_FEQ(r, b, 1e-6);
                }
            }
        }
    }

error:
    pl_dispatch_destroy(&dp);
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &src_tex);
    pl_tex_destroy(gpu, &lsb_tex);
    pl_tex_destroy(gpu, &msb_tex);
    pl_tex_destroy(gpu, &fbo_tex);
}

// The deband grain is relative to the reference white of the image. An SDR
// signal peaks at its white, whatever luminance that is tagged with, so the
// grain must not depend on it. An HDR signal gets the grain scaled down by its
// peak relative to the reference white.
static void pl_render_deband_tests(pl_gpu gpu)
{
    pl_renderer rr = NULL;
    pl_tex src_tex = NULL, fbo = NULL;
    printf("pl_render_deband_tests:\n");

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                             PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fmt) {
        printf("- no suitable f32 fmt, skipping\n");
        return;
    }

    enum { width = 64, height = 64 };
    static const float gray = 0.5f;

    // No alpha, the image is not to be blended against anything
    static float src_data[height][width][4];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++)
                src_data[y][x][c] = gray;
        }
    }

    struct pl_plane src_plane = {0};
    struct pl_plane_data src_pdata = {
        .type           = PL_FMT_FLOAT,
        .width          = width,
        .height         = height,
        .component_size = { 32, 32, 32 },
        .component_map  = { 0, 1, 2 },
        .pixel_stride   = 4 * sizeof(float),
        .pixels         = src_data,
    };
    if (!pl_upload_plane(gpu, &src_plane, &src_tex, &src_pdata))
        goto error;

    fbo = pl_tex_create(gpu, pl_tex_params(
        .w = width, .h = height, .format = fmt,
        .renderable = true, .host_readable = true,
    ));
    if (!fbo)
        goto error;

    rr = pl_renderer_create(gpu->log, gpu);
    REQUIRE(rr);

    static const struct {
        enum pl_color_transfer trc;
        float image_luma, target_luma;  // tagged peak in cd/m^2, 0 if untagged
        float scale;                    // expected grain, 1 is the configured one
    } tests[] = {
        { PL_COLOR_TRC_SRGB,       0,    0, 1.0f },
        { PL_COLOR_TRC_SRGB,       0,  100, 1.0f },
        { PL_COLOR_TRC_SRGB,      80,    0, 1.0f },
        { PL_COLOR_TRC_SRGB,     480,  480, 1.0f },
        { PL_COLOR_TRC_BT_1886,  100,    0, 1.0f },
        { PL_COLOR_TRC_GAMMA22,  300,  300, 1.0f },
        { PL_COLOR_TRC_LINEAR,    80,    0, 1.0f },
        { PL_COLOR_TRC_PQ,         0,    0, PL_COLOR_SDR_WHITE / 10000 },
        { PL_COLOR_TRC_PQ,      1000, 1000, PL_COLOR_SDR_WHITE / 1000 },
        { PL_COLOR_TRC_HLG,        0,    0, PL_COLOR_SDR_WHITE / PL_COLOR_HLG_PEAK },
        { PL_COLOR_TRC_HLG,      400,  400, PL_COLOR_SDR_WHITE / 400 },
    };

    // Peak to peak amplitude of the grain. The grain is configured to come
    // out the same in every test, so that they are equally precise.
    static const float amplitude = 0.2f;

    for (int i = 0; i < PL_ARRAY_SIZE(tests); i++) {
        struct pl_frame image = {
            .num_planes = 1,
            .planes     = { src_plane },
            .repr       = pl_color_repr_rgb,
            .color      = {
                .primaries    = PL_COLOR_PRIM_BT_709,
                .transfer     = tests[i].trc,
                .hdr.max_luma = tests[i].image_luma,
            },
        };

        // Same color space, to get the image out the way it went in
        struct pl_frame target = {
            .num_planes = 1,
            .planes     = {{
                .texture           = fbo,
                .components        = 3,
                .component_mapping = {0, 1, 2},
            }},
            .repr       = pl_color_repr_rgb,
            .color      = image.color,
        };
        target.color.hdr.max_luma = tests[i].target_luma;

        struct pl_deband_params deband = pl_deband_default_params;
        deband.grain = 1000 * amplitude / tests[i].scale;

        printf("- %s image at %g cd/m^2, target at %g cd/m^2\n",
               pl_color_transfer_name(tests[i].trc), tests[i].image_luma,
               tests[i].target_luma);

        static float out[height][width][4];
        for (int grain = 0; grain < 2; grain++) {
            struct pl_render_params params = {
                .deband_params = grain ? &deband : NULL,
            };
            REQUIRE(pl_render_image(rr, &image, &target, &params));
            REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
            REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
                .tex = fbo,
                .ptr = out,
            )));

            float min = gray, max = gray;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < 3; c++) {
                        REQUIRE(isfinite(out[y][x][c]));
                        min = PL_MIN(min, out[y][x][c]);
                        max = PL_MAX(max, out[y][x][c]);
                    }
                }
            }

            if (grain) {
                REQUIRE_FEQ((max - min) / amplitude, 1.0, 1e-2);
            } else {
                REQUIRE_FEQ(min, gray, 1e-6);
                REQUIRE_FEQ(max, gray, 1e-6);
            }
        }
    }

error:
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &src_tex);
    pl_tex_destroy(gpu, &fbo);
}

static struct pl_hook_res noop_hook(void *priv, const struct pl_hook_params *params)
{
    return (struct pl_hook_res) {0};
}

static void pl_ycbcr_tests(pl_gpu gpu)
{
    const int depth = 16;
    struct pl_plane_data data[3];
    for (int i = 0; i < 3; i++) {
        const int sub = i > 0 ? 1 : 0;
        const int width = (323 + sub) >> sub;
        const int height = (255 + sub) >> sub;

        data[i] = (struct pl_plane_data) {
            .type = PL_FMT_UNORM,
            .width = width,
            .height = height,
            .component_size = {depth},
            .component_map = {i},
            .pixel_stride = sizeof(uint16_t),
            .row_stride = PL_ALIGN2(width * sizeof(uint16_t),
                                    gpu->limits.align_tex_xfer_pitch),
        };
    }

    pl_fmt fmt = pl_plane_find_fmt(gpu, NULL, &data[0]);
    enum pl_fmt_caps caps = PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE;
    if (!fmt || (fmt->caps & caps) != caps)
        return;

    printf("pl_ycbcr_tests:\nfmt=%s\n", fmt->name);

    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    if (!rr)
        return;

    pl_tex src_tex[3] = {0};
    pl_tex dst_tex[3] = {0};
    struct pl_frame img = {
        .num_planes = 3,
        .repr = pl_color_repr_hdtv,
        .color = pl_color_space_bt709,
    };

    struct pl_frame target = {
        .num_planes = 3,
        .repr = pl_color_repr_hdtv,
        .color = pl_color_space_bt709,
    };

    uint8_t *src_buffer[3] = {0};
    uint8_t *dst_buffer = NULL;
    for (int i = 0; i < 3; i++) {
        // Generate some arbitrary data for the buffer
        src_buffer[i] = malloc(data[i].height * data[i].row_stride);
        if (!src_buffer[i])
            goto error;

        data[i].pixels = src_buffer[i];
        for (int y = 0; y < data[i].height; y++) {
            for (int x = 0; x < data[i].width; x++) {
                size_t off = y * data[i].row_stride + x * data[i].pixel_stride;
                uint16_t *pixel = (uint16_t *) &src_buffer[i][off];
                int gx = 200 + 100 * i, gy = 300 + 150 * i;
                *pixel = (gx * x) ^ (gy * y); // whatever
            }
        }

        REQUIRE(pl_upload_plane(gpu, &img.planes[i], &src_tex[i], &data[i]));
    }

    // This co-sites chroma pixels with pixels in the RGB image, meaning we
    // get an exact round-trip when sampling both ways. This makes it useful
    // as a test case, even though it's not common in the real world.
    pl_frame_set_chroma_location(&img, PL_CHROMA_TOP_LEFT);

    for (int i = 0; i < 3; i++) {
        dst_tex[i] = pl_tex_create(gpu, &(struct pl_tex_params) {
            .format = fmt,
            .w = data[i].width,
            .h = data[i].height,
            .renderable = true,
            .host_readable = true,
            .storable = fmt->caps & PL_FMT_CAP_STORABLE,
            .blit_dst = fmt->caps & PL_FMT_CAP_BLITTABLE,
        });

        if (!dst_tex[i])
            goto error;

        target.planes[i] = img.planes[i];
        target.planes[i].texture = dst_tex[i];
    }

    REQUIRE(pl_render_image(rr, &img, &target, &(struct pl_render_params) {
        .num_hooks = 1,
        .hooks = &(const struct pl_hook *){&(struct pl_hook) {
            // Forces chroma merging, to test the chroma merging code
            .stages = PL_HOOK_CHROMA_INPUT,
            .hook = noop_hook,
        }},
    }));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);

    size_t buf_size = data[0].height * data[0].row_stride;
    dst_buffer = malloc(buf_size);
    if (!dst_buffer)
        goto error;

    for (int i = 0; i < 3; i++) {
        memset(dst_buffer, 0xAA, buf_size);
        REQUIRE(pl_tex_download(gpu, &(struct pl_tex_transfer_params) {
            .tex = dst_tex[i],
            .ptr = dst_buffer,
            .row_pitch = data[i].row_stride,
        }));

        for (int y = 0; y < data[i].height; y++) {
            for (int x = 0; x < data[i].width; x++) {
                size_t off = y * data[i].row_stride + x * data[i].pixel_stride;
                uint16_t *src_pixel = (uint16_t *) &src_buffer[i][off];
                uint16_t *dst_pixel = (uint16_t *) &dst_buffer[off];
                // Test that the dst_pixel is correctly clamped to the
                // legal signal range
                const uint16_t min_val = 16 << (depth - 8);
                const uint16_t max_val = (i ? 240 : 235) << (depth - 8);
                uint16_t src_clamped = PL_CLAMP(*src_pixel, min_val, max_val);
                int diff = abs((int) src_clamped - (int) *dst_pixel);
                REQUIRE_CMP(diff, <=, 150, "d"); // a little over 0.2%
            }
        }
    }

error:
    pl_renderer_destroy(&rr);
    free(dst_buffer);
    for (int i = 0; i < 3; i++) {
        free(src_buffer[i]);
        pl_tex_destroy(gpu, &src_tex[i]);
        pl_tex_destroy(gpu, &dst_tex[i]);
    }
}

static void pl_test_export_import(pl_gpu gpu,
                                  enum pl_handle_type handle_type)
{
    printf("pl_test_export_import:\n");
    // Test texture roundtrip

    if (!(gpu->export_caps.tex & handle_type) ||
        !(gpu->import_caps.tex & handle_type))
        goto skip_tex;

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 0, 0, PL_FMT_CAP_BLITTABLE);
    if (!fmt || (handle_type == PL_HANDLE_DMA_BUF && !fmt->num_modifiers))
        goto skip_tex;

    printf("- testing texture import/export with fmt %s\n", fmt->name);

    pl_tex export = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = 32,
        .h = 32,
        .format = fmt,
        .export_handle = handle_type,
    });
    REQUIRE(export);
    REQUIRE_HANDLE(export->shared_mem, handle_type);

    pl_tex import = pl_tex_create(gpu, &(struct pl_tex_params) {
        .w = export->params.w,
        .h = export->params.h,
        .format = fmt,
        .import_handle = handle_type,
        .shared_mem = export->shared_mem,
    });
    REQUIRE(import);

    pl_tex_destroy(gpu, &import);
    pl_tex_destroy(gpu, &export);

skip_tex: ;

    // Test buffer roundtrip

    if (!(gpu->export_caps.buf & handle_type) ||
        !(gpu->import_caps.buf & handle_type))
        return;

    printf("- testing buffer import/export\n");

    pl_buf exp_buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = 32,
        .export_handle = handle_type,
    });
    REQUIRE(exp_buf);
    REQUIRE_HANDLE(exp_buf->shared_mem, handle_type);

    pl_buf imp_buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = 32,
        .import_handle = handle_type,
        .shared_mem = exp_buf->shared_mem,
    });
    REQUIRE(imp_buf);

    pl_buf_destroy(gpu, &imp_buf);
    pl_buf_destroy(gpu, &exp_buf);
}

static void pl_test_host_ptr(pl_gpu gpu)
{
    if (!(gpu->import_caps.buf & PL_HANDLE_HOST_PTR))
        return;

    printf("pl_test_host_ptr:\n");

#ifdef __unix__

    printf("- testing host ptr\n");
    REQUIRE(gpu->limits.max_mapped_size);

    const size_t size = 2 << 20;
    const size_t offset = 2 << 10;
    const size_t slice = 2 << 16;

    uint8_t *data = aligned_alloc(0x1000, size);
    for (int i = 0; i < size; i++)
        data[i] = (uint8_t) i;

    pl_buf buf = pl_buf_create(gpu, &(struct pl_buf_params) {
        .size = slice,
        .import_handle = PL_HANDLE_HOST_PTR,
        .shared_mem = {
            .handle.ptr = data,
            .size = size,
            .offset = offset,
        },
        .host_mapped = true,
    });

    REQUIRE(buf);
    REQUIRE_MEMEQ(data + offset, buf->data, slice);

    pl_buf_destroy(gpu, &buf);
    free(data);

#endif // unix
}

static void pl_render_sampling_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 16,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_LINEAR | PL_FMT_CAP_RENDERABLE |
        PL_FMT_CAP_HOST_READABLE);
    if (!fmt)
        return;

    enum { W = 128, H = 96 };
    uint16_t data[W * H * 4], out[W * H * 4];
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < 3; c++)
                data[(y * W + x) * 4 + c] = (x + y + c) % 2 ? 0x3c00 : 0;
            data[(y * W + x) * 4 + 3] = 0x3c00;
        }
    }

    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .sampleable = true, .initial_data = data,
    ));
    pl_tex dst = NULL;
    REQUIRE(src);
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    for (int test = 0; test < 4; test++) {
        const bool cropped = test == 1 || test == 2;
        const int width = cropped ? 64 : test == 3 ? W / 2 : W;
        const int height = cropped ? 64 : test == 3 ? H / 2 : H;
        const float offset = test == 2 ? 0.25f : 0.0f;
        printf("- testing float sampling, case %d, offset %.2f\n", test, offset);
        REQUIRE(pl_tex_recreate(gpu, &dst, pl_tex_params(
            .w = width, .h = height, .format = fmt,
            .renderable = true, .host_readable = true,
        )));
        struct pl_frame image = {
            .num_planes = 1,
            .planes[0] = { .texture = src, .components = 4,
                           .component_mapping = {0, 1, 2, 3} },
            .repr = pl_color_repr_rgb,
            .color = pl_color_space_bt709,
            .crop = { cropped ? 32 + offset : 0, cropped ? 16 + offset : 0,
                      cropped ? 96 + offset : W, cropped ? 80 + offset : H },
        };
        image.color.transfer = PL_COLOR_TRC_GAMMA22;
        struct pl_frame target = image;
        target.planes[0].texture = dst;
        target.crop = (pl_rect2df) { 0, 0, width, height };
        struct pl_render_params params = pl_render_fast_params;
        params.upscaler = &pl_filter_bilinear;
        params.correct_subpixel_offsets = true;
        REQUIRE(pl_render_image(rr, &image, &target, &params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = out)));
        for (int y = 0; y < height - (test == 2); y++) {
            for (int x = 0; x < width - (test == 2); x++) {
                for (int c = 0; c < 4; c++) {
                    uint16_t expect = c == 3 || (x + y + c) % 2 ? 0x3c00 : 0;
                    if (test == 2 && c != 3)
                        expect = expect ? 0x3900 : 0x3600; // 0.625 or 0.375
                    if (test == 3 && c != 3)
                        expect = 0x3800; // 0.5
                    REQUIRE_CMP(out[(y * width + x) * 4 + c], ==, expect, "u");
                }
            }
        }
        REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_NONE);
    }
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &dst);
}

// At 1:1 scale, the renderer drops sub-pixel offsets from the source crop,
// so these must round to the nearest pixel. In particular, float error just
// below a whole pixel must not shift the image by an entire pixel.
static void pl_render_offset_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_named_fmt(gpu, "rgba8");
    if (!fmt || !(fmt->caps & PL_FMT_CAP_RENDERABLE) ||
        !(fmt->caps & PL_FMT_CAP_HOST_READABLE))
        return;

    enum { W = 16, N = 8 };
    uint8_t data[W * 4];
    for (int x = 0; x < W; x++) {
        data[x * 4 + 0] = x * 16;
        data[x * 4 + 1] = 255 - x * 16;
        data[x * 4 + 2] = 0;
        data[x * 4 + 3] = 255;
    }

    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = 1, .format = fmt, .sampleable = true, .initial_data = data,
    ));
    pl_tex dst = pl_tex_create(gpu, pl_tex_params(
        .w = N, .h = 1, .format = fmt, .renderable = true, .host_readable = true,
    ));
    REQUIRE(src && dst);
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);

    static const float offsets[] = { 3.99999976f, 4.0f, 3.6f, 4.4f };
    for (int i = 0; i < PL_ARRAY_SIZE(offsets); i++) {
        const float x0 = offsets[i];
        printf("- testing source crop offset %.8f at 1:1 scale\n", x0);
        struct pl_frame image = {
            .num_planes = 1,
            .planes[0]  = { .texture = src, .components = 4,
                            .component_mapping = {0, 1, 2, 3} },
            .repr       = pl_color_repr_rgb,
            .color      = pl_color_space_srgb,
            .crop       = { x0, 0, x0 + N, 1 },
        };
        struct pl_frame target = image;
        target.planes[0].texture = dst;
        target.crop = (pl_rect2df) { 0, 0, N, 1 };

        REQUIRE(pl_render_image(rr, &image, &target, &pl_render_fast_params));
        uint8_t out[N * 4];
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = out)));
        for (int x = 0; x < N; x++) {
            // Expect texel 4 + x, the nearest to every offset tested
            REQUIRE_CMP(abs(out[x * 4 + 0] - data[(4 + x) * 4 + 0]), <=, 1, "d");
            REQUIRE_CMP(abs(out[x * 4 + 1] - data[(4 + x) * 4 + 1]), <=, 1, "d");
        }
    }

    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &dst);
}

#ifdef PL_HAVE_DOVI

// The composer of ETSI GS CCM 001 V1.1.1 clause 5.4, in its integer arithmetic
struct dovi_ref_curve {
    int num_pivots;
    int pivots[9];
    int mapping_idc[8]; // 0: polynomial, 1: MMR
    int poly_order[8];
    int64_t poly_coef[8][3];
    int mmr_order[8];
    int64_t mmr_constant[8];
    int64_t mmr_coef[8][3][7];
};

struct dovi_ref {
    int bl_bit_depth, el_bit_depth, coef_log2_denom;
    struct dovi_ref_curve curve[3];
    struct {
        int offset;
        int64_t slope, threshold, hdr_in_max;
    } nlq[3];
};

// 5.4.2.2, taking the last piece for a sample at or above the last pivot
static int dovi_ref_piece(const struct dovi_ref_curve *cv, int s)
{
    for (int i = 0; i < cv->num_pivots - 1; i++) {
        if (s < cv->pivots[i + 1])
            return i;
    }
    return cv->num_pivots - 2;
}

static int64_t dovi_ref_clamp(const struct dovi_ref_curve *cv, int s)
{
    return PL_CLAMP(s, cv->pivots[0], cv->pivots[cv->num_pivots - 1]);
}

// 5.4.2.3.2
static int64_t dovi_ref_poly(const struct dovi_ref *ref, int cmp, int piece, int s)
{
    const struct dovi_ref_curve *cv = &ref->curve[cmp];
    const int64_t sc = dovi_ref_clamp(cv, s);
    int64_t ss = 1, vv = 0;
    int shift = 20;
    for (int i = 0; i <= cv->poly_order[piece]; i++) {
        vv += cv->poly_coef[piece][i] * (ss << shift);
        ss *= sc;
        shift -= ref->bl_bit_depth;
    }
    vv = PL_MAX(vv, 0);
    return PL_MIN(vv >> (4 + ref->coef_log2_denom), 0xffff);
}

// 5.4.2.3.3, with the luma of the sample itself as the down-sampled luma
static int64_t dovi_ref_mmr(const struct dovi_ref *ref, int cmp, int piece,
                            const int sample[3])
{
    const struct dovi_ref_curve *cv = &ref->curve[cmp];
    const int n = ref->bl_bit_depth;
    int64_t s[3];
    for (int c = 0; c < 3; c++)
        s[c] = dovi_ref_clamp(&ref->curve[c], sample[c]);

    int64_t tt[22];
    tt[0] = 1 << 20;
    tt[1] = s[0] << (20 - n);
    tt[2] = s[1] << (20 - n);
    tt[3] = s[2] << (20 - n);
    tt[4] = (s[0] * s[1]) << (20 - 2 * n);
    tt[5] = (s[0] * s[2]) << (20 - 2 * n);
    tt[6] = (s[1] * s[2]) << (20 - 2 * n);
    tt[7] = (tt[4] * tt[3]) >> 20;
    tt[8] = (s[0] * s[0]) << (20 - 2 * n);
    tt[9] = (s[1] * s[1]) << (20 - 2 * n);
    tt[10] = (s[2] * s[2]) << (20 - 2 * n);
    tt[11] = (tt[4] * tt[4]) >> 20;
    tt[12] = (tt[5] * tt[5]) >> 20;
    tt[13] = (tt[6] * tt[6]) >> 20;
    tt[14] = (tt[7] * tt[7]) >> 20;
    tt[15] = (tt[1] * tt[8]) >> 20;
    tt[16] = (tt[2] * tt[9]) >> 20;
    tt[17] = (tt[3] * tt[10]) >> 20;
    tt[18] = (tt[4] * tt[11]) >> 20;
    tt[19] = (tt[5] * tt[12]) >> 20;
    tt[20] = (tt[6] * tt[13]) >> 20;
    tt[21] = (tt[7] * tt[14]) >> 20;

    int64_t rr = cv->mmr_constant[piece] * tt[0];
    int cnt = 1;
    for (int i = 0; i < cv->mmr_order[piece]; i++) {
        for (int j = 0; j < 7; j++)
            rr += cv->mmr_coef[piece][i][j] * tt[cnt++];
    }
    rr = PL_MAX(rr, 0);
    return PL_MIN(rr >> (4 + ref->coef_log2_denom), 0xffff);
}

// 5.4.3.2
static int64_t dovi_ref_nlq(const struct dovi_ref *ref, int cmp, int e)
{
    const int64_t T = ref->nlq[cmp].threshold, S = ref->nlq[cmp].slope,
                  R = ref->nlq[cmp].hdr_in_max;
    const int eld = ref->el_bit_depth;
    int64_t rr = e - ref->nlq[cmp].offset;
    if (rr == 0)
        return 0;

    const int64_t sign = rr < 0 ? -1 : 1;
    rr = (2 * rr - sign) * (1 << (10 - eld));
    int64_t dq = rr * S + (T << (10 - eld + 1)) * sign;
    const int64_t RR = R << (10 - eld + 1);
    dq = PL_CLAMP(dq, -RR, RR);

    // arithmetic shift, rounding towards minus infinity
    const int64_t div = (int64_t) 1 << (ref->coef_log2_denom - 5 - eld);
    int64_t r = dq / div;
    if (dq % div && dq < 0)
        r--;
    return r;
}

// The composed signal h of 5.4.3.3 of component `cmp`, before its rounding to
// hdr_bit_depth, as a fraction of 2^16. At an interior pivot the float
// rounding of the sample may select either piece, so both are returned.
static int dovi_ref_compose(const struct dovi_ref *ref, int cmp, const int bl[3],
                            int el, double out[2], bool *mmr)
{
    const struct dovi_ref_curve *cv = &ref->curve[cmp];
    const int piece = dovi_ref_piece(cv, bl[cmp]);
    const int64_t r = dovi_ref_nlq(ref, cmp, el);
    int num = 0;
    *mmr = false;
    for (int p = piece; p >= PL_MAX(piece - 1, 0); p--) {
        if (p < piece && bl[cmp] != cv->pivots[piece])
            break;
        int64_t v = cv->mapping_idc[p] ? dovi_ref_mmr(ref, cmp, p, bl)
                                       : dovi_ref_poly(ref, cmp, p, bl[cmp]);
        *mmr |= cv->mapping_idc[p] == 1;
        out[num++] = PL_CLAMP((v + r) / 65536.0, 0.0, 1.0);
    }
    return num;
}

// SMPTE ST 2084, relative to 10000 cd/m^2
static double dovi_pq_eotf(double x)
{
    const double m1 = 2610.0 / 16384, m2 = 2523.0 / 32, c1 = 3424.0 / 4096,
                 c2 = 2413.0 / 128, c3 = 2392.0 / 128;
    const double y = pow(PL_CLAMP(x, 0.0, 1.0), 1.0 / m2);
    return pow(fmax(y - c1, 0.0) / (c2 - c3 * y), 1.0 / m1);
}

static double dovi_pq_oetf(double lin)
{
    const double m1 = 2610.0 / 16384, m2 = 2523.0 / 32, c1 = 3424.0 / 4096,
                 c2 = 2413.0 / 128, c3 = 2392.0 / 128;
    const double y = pow(fmax(lin, 0.0), m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

// The decoding of the composed signal by pl_shader_decode_color: the PQ EOTF,
// the matrix `mat` and the PQ OETF
static void dovi_ref_decode(const pl_matrix3x3 *mat, const double h[3], double out[3])
{
    double lin[3];
    for (int c = 0; c < 3; c++)
        lin[c] = dovi_pq_eotf(h[c]);
    for (int c = 0; c < 3; c++) {
        double sum = 0.0;
        for (int k = 0; k < 3; k++)
            sum += mat->m[c][k] * lin[k];
        out[c] = dovi_pq_oetf(sum);
    }
}

#define Q23(x) ((int64_t) llround((x) * (double) (1 << 23)))

// Mappings of all kinds, with pivots inside the code range and a residual
// bound that bites often (luma), at the largest codes only (Cr) or never (Cb),
// given for 10-bit codes and shifted down for 8-bit ones
static void dovi_test_ref(int depth, struct dovi_ref *ref)
{
    const int sh = 10 - depth;
    *ref = (struct dovi_ref) {
        .bl_bit_depth = depth,
        .el_bit_depth = depth,
        .coef_log2_denom = 23,
        .curve = {
            {
                .num_pivots = 5,
                .pivots = {64 >> sh, 128 >> sh, 300 >> sh, 700 >> sh, 940 >> sh},
                .mapping_idc = {0, 0, 0, 0},
                .poly_order = {1, 2, 2, 1},
                .poly_coef = {
                    {Q23(0.02), Q23(0.9), 0},
                    {Q23(-0.05), Q23(1.6), Q23(-1.2)},
                    {Q23(0.157), Q23(0.636), Q23(-0.113)},
                    {Q23(-0.3), Q23(1.5), 0},
                },
            }, {
                .num_pivots = 3,
                .pivots = {0, 512 >> sh, 1023 >> sh},
                .mapping_idc = {1, 1},
                .mmr_order = {2, 3},
                .mmr_constant = {Q23(0.3), Q23(0.2)},
                .mmr_coef = {
                    {
                        {Q23(0.35), Q23(0.6), Q23(0.08), Q23(-0.45), Q23(-0.2), Q23(-0.3), Q23(0.25)},
                        {Q23(-0.12), Q23(-0.4), Q23(-0.05), Q23(0.3), Q23(0.2), Q23(0.15), Q23(-0.3)},
                    }, {
                        {Q23(0.3), Q23(0.5), Q23(0.2), Q23(-0.3), Q23(-0.25), Q23(0.1), Q23(0.15)},
                        {Q23(-0.1), Q23(-0.2), Q23(-0.15), Q23(0.25), Q23(0.2), Q23(-0.05), Q23(-0.1)},
                        {Q23(0.12), Q23(0.08), Q23(0.1), Q23(-0.15), Q23(-0.1), Q23(0.05), Q23(0.2)},
                    },
                },
            }, {
                .num_pivots = 3,
                .pivots = {0, 400 >> sh, 1023 >> sh},
                .mapping_idc = {0, 1},
                .poly_order = {2},
                .poly_coef = {{Q23(0.5), Q23(0.3), Q23(-0.2)}},
                .mmr_order = {0, 1},
                .mmr_constant = {0, Q23(0.1)},
                .mmr_coef = {
                    {{0}},
                    {{Q23(0.2), Q23(0.45), Q23(0.35), Q23(-0.15), Q23(0.1), Q23(-0.3), Q23(0.5)}},
                },
            },
        },
        .nlq = {
            {.offset = 512 >> sh, .slope = 1 << 15, .threshold = 1 << 13, .hdr_in_max = 1 << 17},
            {.offset = 512 >> sh, .slope = 3 << 13, .threshold = 1 << 12, .hdr_in_max = 1 << 19},
            {.offset = 512 >> sh, .slope = 5 << 12, .threshold = 0,       .hdr_in_max = 1 << 16},
        },
    };
}

// The same metadata as pl_map_dovi_metadata() makes of an RPU, with the
// inverse of the LMS to RGB matrix of pl_shader_decode_color as the RGB to
// LMS matrix, so that the decoding returns the composed signal up to the
// float rounding of their product, which is returned in `mat`
static void dovi_test_metadata(const struct dovi_ref *ref,
                               struct pl_dovi_metadata *out, pl_matrix3x3 *mat)
{
    const double coef = ldexp(1.0, -ref->coef_log2_denom);
    const double pivot = ldexp(1.0, -ref->bl_bit_depth);

    const pl_matrix3x3 lms2rgb = {{
        { 3.064396767855625,    -2.165961226711146,     0.1015574148869036},
        {-0.65613130540261551,   1.7855694129834121,   -0.12943951030147041},
        { 0.017360633500538487, -0.047244531564539943,  1.0298894699757705},
    }};
    pl_matrix3x3 rgb2lms = lms2rgb;
    pl_matrix3x3_invert(&rgb2lms);
    *mat = lms2rgb;
    pl_matrix3x3_mul(mat, &rgb2lms);

    *out = (struct pl_dovi_metadata) {
        .nonlinear  = pl_matrix3x3_identity,
        .linear     = rgb2lms,
        .nlq_active = true,
    };

    for (int c = 0; c < 3; c++) {
        const struct dovi_ref_curve *cv = &ref->curve[c];
        struct pl_reshape_data *comp = &out->comp[c];
        comp->num_pivots = cv->num_pivots;
        for (int i = 0; i < cv->num_pivots; i++)
            comp->pivots[i] = pivot * cv->pivots[i];
        for (int i = 0; i < cv->num_pivots - 1; i++) {
            comp->method[i] = cv->mapping_idc[i];
            if (cv->mapping_idc[i] == 0) {
                for (int k = 0; k <= cv->poly_order[i]; k++)
                    comp->poly_coeffs[i][k] = coef * cv->poly_coef[i][k];
            } else {
                comp->mmr_order[i] = cv->mmr_order[i];
                comp->mmr_constant[i] = coef * cv->mmr_constant[i];
                for (int j = 0; j < cv->mmr_order[i]; j++) {
                    for (int k = 0; k < 7; k++)
                        comp->mmr_coeffs[i][j][k] = coef * cv->mmr_coef[i][j][k];
                }
            }
        }

        out->nlq[c].offset = ref->nlq[c].offset;
        out->nlq[c].deadzone_slope = coef * ref->nlq[c].slope;
        out->nlq[c].deadzone_threshold = coef * ref->nlq[c].threshold;
        out->nlq[c].hdr_in_max = coef * ref->nlq[c].hdr_in_max;
    }
}

// Every luma code, with chroma and enhancement layer codes that differ per
// sample, the latter on both sides of the offset and at it
static void dovi_test_codes(int x, int y, int depth, int offset, int bl[3], int el[3])
{
    const int max = (1 << depth) - 1;
    bl[0] = x;
    bl[1] = (x * 7 + y * 131) & max;
    bl[2] = (x * 13 + y * 257 + 100) & max;
    el[0] = PL_CLAMP(offset + (x + y) % 61 - 30, 0, max);
    el[1] = PL_CLAMP(offset + (x * 3 + y) % 23 - 11, 0, max);
    el[2] = PL_CLAMP(offset + (x * 5 + y * 7) % 9 - 4, 0, max);
}

static void pl_dovi_tests(pl_gpu gpu)
{
    pl_fmt fbo_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                                 PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fbo_fmt)
        return;
    pl_fmt f32_fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
                                 PL_FMT_CAP_SAMPLEABLE);
    printf("pl_dovi_tests:\n");

    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(dp);

    static const struct {
        int depth, tex_bits, shift;
        enum pl_color_levels levels;
        bool normalized; // by the renderer, into a float texture
    } variants[] = {
        {10, 16, 0, PL_COLOR_LEVELS_FULL},
        {10, 16, 0, PL_COLOR_LEVELS_LIMITED},
        {10, 16, 6, PL_COLOR_LEVELS_LIMITED},
        { 8,  8, 0, PL_COLOR_LEVELS_FULL},
        { 8, 16, 0, PL_COLOR_LEVELS_LIMITED},
        { 8, 16, 8, PL_COLOR_LEVELS_FULL},
        {10, 16, 0, PL_COLOR_LEVELS_FULL,    true},
        {10, 16, 6, PL_COLOR_LEVELS_LIMITED, true},
        { 8, 16, 8, PL_COLOR_LEVELS_FULL,    true},
    };

    for (int i = 0; i < PL_ARRAY_SIZE(variants); i++) {
        const int depth = variants[i].depth, tex_bits = variants[i].tex_bits,
                  shift = variants[i].shift;
        const bool normalized = variants[i].normalized;
        pl_fmt tex_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, tex_bits, tex_bits,
                                     PL_FMT_CAP_SAMPLEABLE);
        if (!tex_fmt || (normalized && !f32_fmt)) {
            printf("- no %d-bit texture format, skipping\n",
                   tex_fmt ? 32 : tex_bits);
            continue;
        }

        struct dovi_ref ref;
        dovi_test_ref(depth, &ref);
        struct pl_dovi_metadata meta;
        pl_matrix3x3 mat;
        dovi_test_metadata(&ref, &meta, &mat);

        struct pl_color_repr repr = {
            .sys    = PL_COLOR_SYSTEM_DOLBYVISION,
            .levels = variants[i].levels,
            .dovi   = &meta,
            .bits   = {
                .sample_depth = tex_bits,
                .color_depth  = depth,
                .bit_shift    = shift,
            },
        };

        // The renderer normalizes every plane before the decode
        const float norm = normalized ? pl_color_repr_normalize(&repr) : 1.0f;
        if (normalized)
            REQUIRE_CMP(repr.bits.sample_depth, ==, depth, "d");

        const int W = 1 << depth, H = 8;
        uint16_t *bl_data = malloc(W * H * 4 * sizeof(uint16_t));
        uint16_t *el_data = malloc(W * H * 4 * sizeof(uint16_t));
        float *bl_float = malloc(W * H * 4 * sizeof(float));
        uint8_t *bl_data8 = (uint8_t *) bl_data, *el_data8 = (uint8_t *) el_data;
        REQUIRE(bl_data && el_data && bl_float);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int bl[3], el[3];
                dovi_test_codes(x, y, depth, ref.nlq[0].offset, bl, el);
                for (int c = 0; c < 4; c++) {
                    const int bv = c < 3 ? bl[c] << shift : 0,
                              ev = c < 3 ? el[c] << shift : 0;
                    if (tex_bits == 16) {
                        bl_data[(y * W + x) * 4 + c] = bv;
                        el_data[(y * W + x) * 4 + c] = ev;
                    } else {
                        bl_data8[(y * W + x) * 4 + c] = bv;
                        el_data8[(y * W + x) * 4 + c] = ev;
                    }
                    bl_float[(y * W + x) * 4 + c] = norm * bv /
                                                    (float) ((1 << tex_bits) - 1);
                }
            }
        }

        pl_tex bl_tex = pl_tex_create(gpu, pl_tex_params(
            .format = normalized ? f32_fmt : tex_fmt,
            .w = W,
            .h = H,
            .sampleable = true,
            .initial_data = normalized ? (void *) bl_float : (void *) bl_data,
        ));
        pl_tex el_tex = pl_tex_create(gpu, pl_tex_params(
            .format = tex_fmt,
            .w = W,
            .h = H,
            .sampleable = true,
            .initial_data = el_data,
        ));
        pl_tex fbo = pl_tex_create(gpu, pl_tex_params(
            .format = fbo_fmt,
            .w = W,
            .h = H,
            .renderable = true,
            .host_readable = true,
        ));
        REQUIRE(bl_tex && el_tex && fbo);

        // The enhancement layer sub-shader returns the EL codes
        pl_shader el_sh = pl_dispatch_begin_ex(dp, true);
        pl_shader_sample_nearest(el_sh, pl_sample_src(
            .tex   = el_tex,
            .scale = ldexpf((1 << tex_bits) - 1, -shift),
        ));

        pl_shader sh = pl_dispatch_begin_ex(dp, true);
        pl_shader_sample_nearest(sh, pl_sample_src( .tex = bl_tex ));
        pl_shader_decode_color_ex(sh, pl_color_decode_args(
            .repr              = &repr,
            .enhancement_layer = el_sh,
        ));
        pl_dispatch_abort(dp, &el_sh);
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &sh,
            .target = fbo,
        )));

        float *out = malloc(W * H * 4 * sizeof(float));
        REQUIRE(out);
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
            .tex = fbo,
            .ptr = out,
        )));

        // The specification truncates the mapped signal and the residual to
        // integers of 2^16 and the MMR terms to 2^20, the shader does not
        int bad = 0;
        double max_err = 0.0;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int bl[3], el[3], num[3];
                double h[3][2];
                bool mmr[3];
                dovi_test_codes(x, y, depth, ref.nlq[0].offset, bl, el);
                for (int c = 0; c < 3; c++)
                    num[c] = dovi_ref_compose(&ref, c, bl, el[c], h[c], &mmr[c]);

                // The closest of the decodings of all candidate signals
                double err[3] = { INFINITY, INFINITY, INFINITY };
                for (int k = 0; k < num[0] * num[1] * num[2]; k++) {
                    const double hk[3] = {
                        h[0][k % num[0]],
                        h[1][k / num[0] % num[1]],
                        h[2][k / num[0] / num[1]],
                    };
                    double expect[3];
                    dovi_ref_decode(&mat, hk, expect);
                    for (int c = 0; c < 3; c++) {
                        const float got = out[(y * W + x) * 4 + c];
                        err[c] = fmin(err[c], fabs(got - expect[c]));
                    }
                }

                for (int c = 0; c < 3; c++) {
                    const double tol = mmr[c] ? 1e-4 : 5e-5;
                    if (!(err[c] <= tol)) {
                        if (bad < 5) {
                            printf("  sample (%d, %d) component %d: codes %d %d %d, "
                                   "EL %d: got %f, expected %f\n", x, y, c,
                                   bl[0], bl[1], bl[2], el[c],
                                   out[(y * W + x) * 4 + c], h[c][0]);
                        }
                        bad++;
                    }
                    max_err = fmax(max_err, err[c]);
                }
            }
        }

        printf("- %d-bit codes in %d-bit textures, shift %d, %s range%s: "
               "max error %.2g\n", depth, tex_bits, shift,
               variants[i].levels == PL_COLOR_LEVELS_FULL ? "full" : "limited",
               normalized ? ", normalized" : "", max_err);
        REQUIRE_CMP(bad, ==, 0, "d");

        free(out);
        free(bl_data);
        free(el_data);
        free(bl_float);
        pl_tex_destroy(gpu, &bl_tex);
        pl_tex_destroy(gpu, &el_tex);
        pl_tex_destroy(gpu, &fbo);
    }

    pl_dispatch_destroy(&dp);
}

#endif // PL_HAVE_DOVI

struct custom_mix_test {
    pl_gpu gpu;
    pl_dispatch dispatch;
    pl_tex output;
    int calls, count, reference;
    const double *timestamps;
    uint64_t ids[16];
    float alpha;
    bool fail, blend, verify_clean;
    struct pl_color_space encoded;
};

static pl_tex custom_mix_callback(void *priv, const struct pl_rendered_mix *mix)
{
    struct custom_mix_test *test = priv;
    test->calls++;
    REQUIRE(mix->num_frames == test->count);
    REQUIRE(mix->reference == test->reference);
    REQUIRE(pl_color_transfer_is_normalized_linear(mix->color.transfer));
    for (int i = 0; i < mix->num_frames; i++) {
        const struct pl_rendered_frame *frame = &mix->frames[i];
        REQUIRE(frame->timestamp == test->timestamps[i]);
        REQUIRE(frame->signature);
        REQUIRE(frame->texture->params.format->type == PL_FMT_FLOAT);
        REQUIRE(frame->texture->params.format->num_components == 4);
        for (int c = 0; c < 4; c++)
            REQUIRE(frame->texture->params.format->component_depth[c] == 32);
        test->ids[i] = frame->signature;
    }
    if (test->fail)
        return NULL;
    if (test->verify_clean) {
        pl_tex ref = mix->frames[test->reference].texture;
        REQUIRE(pl_tex_recreate(test->gpu, &test->output, pl_tex_params(
            .w = ref->params.w, .h = ref->params.h, .format = ref->params.format,
            .sampleable = true, .renderable = true, .host_readable = true)));
        pl_shader copy = pl_dispatch_begin(test->dispatch);
        REQUIRE(pl_shader_sample_direct(copy, pl_sample_src(.tex = ref)));
        REQUIRE(pl_dispatch_finish(test->dispatch, pl_dispatch_params(
            .shader = &copy, .target = test->output)));
        size_t count = (size_t) ref->params.w * ref->params.h;
        float *data = malloc(count * 4 * sizeof(float));
        REQUIRE(data);
        REQUIRE(pl_tex_download(test->gpu, pl_tex_transfer_params(
            .tex = test->output, .ptr = data)));
        for (size_t i = 0; i < count; i++) {
            float rgb[] = {0.2f + 0.01f * (i % ref->params.w), 0.3f, 0.4f};
            pl_color_linearize(&test->encoded, rgb);
            for (int c = 0; c < 3; c++)
                REQUIRE(fabsf(data[4*i+c] - rgb[c] * 0.75f) < 0.0001f);
            REQUIRE(fabsf(data[4*i+3] - 0.75f) < 0.000001f);
        }
        free(data);
    }
    if (!test->blend)
        return mix->frames[test->reference].texture;

    pl_tex first = mix->frames[0].texture;
    REQUIRE(pl_tex_recreate(test->gpu, &test->output, pl_tex_params(
        .w = first->params.w, .h = first->params.h,
        .format = first->params.format, .sampleable = true, .renderable = true)));
    struct pl_shader_desc desc[2] = {0};
    for (int i = 0; i < 2; i++) {
        desc[i].desc = (struct pl_desc) {
            .name = i ? "second" : "first", .type = PL_DESC_SAMPLED_TEX,
        };
        desc[i].binding.object = mix->frames[i].texture;
        desc[i].binding.sample_mode = PL_TEX_SAMPLE_NEAREST;
        desc[i].binding.address_mode = PL_TEX_ADDRESS_CLAMP;
    }
    pl_shader sh = pl_dispatch_begin(test->dispatch);
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .input = PL_SHADER_SIG_NONE, .output = PL_SHADER_SIG_COLOR,
        .body = "ivec2 p = ivec2(gl_FragCoord.xy);\n"
                "color = mix(texelFetch(first, p, 0), "
                "texelFetch(second, p, 0), alpha);\n",
        .descriptors = desc, .num_descriptors = 2,
        .variables = &(struct pl_shader_var) {
            .var = pl_var_float("alpha"), .data = &test->alpha,
        },
        .num_variables = 1,
    }));
    REQUIRE(pl_dispatch_finish(test->dispatch, pl_dispatch_params(
        .shader = &sh, .target = test->output)));
    return test->output;
}

static struct pl_frame custom_mix_frame(pl_tex texture)
{
    return (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture = texture, .components = 4,
            .component_mapping = {0, 1, 2, 3},
        },
        .repr = {.sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_FULL,
                 .alpha = PL_ALPHA_INDEPENDENT},
        .color = {
            .primaries = PL_COLOR_PRIM_BT_709, .transfer = PL_COLOR_TRC_GAMMA22,
            .hdr = {.min_luma = 0.1f, .max_luma = 203},
        },
    };
}

static void custom_mix_download(pl_gpu gpu, pl_tex texture, float *data)
{
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
        .tex = texture, .ptr = data)));
}

static void custom_mix_compare(const float *actual, const float *expected, int n)
{
    for (int i = 0; i < n; i++) {
        if (!isfinite(actual[i]) || !isfinite(expected[i]) ||
            !(fabsf(actual[i] - expected[i]) < 0.003f))
        {
            printf("custom mix mismatch index=%d actual=%.9g expected=%.9g "
                   "error=%.9g\n", i, actual[i], expected[i],
                   fabsf(actual[i] - expected[i]));
            fflush(stdout);
        }
        REQUIRE(isfinite(actual[i]) && isfinite(expected[i]));
        REQUIRE(fabsf(actual[i] - expected[i]) < 0.003f);
    }
}

struct custom_mix_owner {
    int acquired, released;
    bool active, set_crop, fail;
    pl_rect2df crop;
    struct pl_overlay *overlay;
    pl_tex overlay_texture;
    const struct pl_frame *next;
};

static bool custom_mix_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    (void) gpu;
    struct custom_mix_owner *owner = frame->user_data;
    REQUIRE(!owner->active);
    owner->active = true;
    owner->acquired++;
    if (owner->next)
        frame->next = owner->next;
    if (owner->set_crop)
        frame->crop = owner->crop;
    if (owner->overlay) {
        owner->overlay->tex = owner->overlay_texture;
        frame->overlays = owner->overlay;
        frame->num_overlays = 1;
    }
    return !owner->fail;
}

static void custom_mix_release(pl_gpu gpu, struct pl_frame *frame)
{
    (void) gpu;
    struct custom_mix_owner *owner = frame->user_data;
    REQUIRE(owner->active);
    owner->active = false;
    owner->released++;
    if (owner->overlay)
        owner->overlay->tex = NULL;
    frame->overlays = NULL;
    frame->num_overlays = 0;
}

enum { SH_W = 32, SH_H = 32, SH_N = SH_W * SH_H * 4 };
struct source_hook_test;
struct source_hook_owner {
    struct source_hook_test *p;
    float pts;
    int acquired, released;
};
struct source_hook_test {
    pl_gpu gpu;
    pl_renderer rr, oracle;
    pl_dispatch dp;
    pl_tex readback, expected;
    const struct pl_hook *hook, *oracle_hook;
    struct pl_frame sources[3];
    struct source_hook_owner owners[3];
    struct pl_render_params params;
    uint64_t ids[2];
    int base, frame0, calls;
    float alpha, gain;
};

static void source_hook_parameter(const struct pl_hook *hook, const char *name, float value)
{
    for (int i = 0; i < hook->num_parameters; i++) {
        if (!strcmp(hook->parameters[i].name, name)) {
            REQUIRE(hook->parameters[i].type == PL_VAR_FLOAT);
            hook->parameters[i].data->f = value;
            return;
        }
    }
    REQUIRE(false);
}

static bool source_hook_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    (void) gpu;
    struct source_hook_owner *o = frame->user_data;
    o->acquired++;
    source_hook_parameter(o->p->hook, "PTS", o->pts);
    return true;
}

static void source_hook_release(pl_gpu gpu, struct pl_frame *frame)
{
    (void) gpu;
    struct source_hook_owner *o = frame->user_data;
    o->released++;
}

static void source_hook_info(void *priv, const struct pl_render_info *event)
{
    struct source_hook_test *p = priv;
    if (event->stage == PL_RENDER_STAGE_FRAME && event->index == 0)
        p->frame0++;
}

static struct pl_frame source_hook_frame(pl_tex texture)
{
    return (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {.texture = texture, .components = 4,
                     .component_mapping = {0, 1, 2, 3}},
        .repr = {.sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_FULL,
                 .alpha = PL_ALPHA_PREMULTIPLIED},
        .color = {.primaries = PL_COLOR_PRIM_BT_709,
                  .transfer = PL_COLOR_TRC_LINEAR_HDR,
                  .hdr = {.min_luma = 0, .max_luma = 203}},
    };
}

static pl_tex source_hook_mix(void *priv, const struct pl_rendered_mix *views)
{
    struct source_hook_test *p = priv;
    REQUIRE(views->num_frames == 2);
    p->calls++;
    for (int i = 0; i < 2; i++) {
        p->ids[i] = views->frames[i].signature;
        REQUIRE(p->ids[i]);
        // Separate renderer and separately parsed parameter storage: the oracle
        // cannot mutate entry/acquire parameter values of the measured renderer.
        struct pl_frame source = p->sources[p->base + i];
        source.acquire = NULL;
        source.release = NULL;
        source_hook_parameter(p->oracle_hook, "PTS", p->owners[p->base + i].pts);
        source_hook_parameter(p->oracle_hook, "gain", p->gain);
        struct pl_frame target = source_hook_frame(p->expected);
        target.color = views->color;
        struct pl_render_params params = p->params;
        params.hooks = &p->oracle_hook;
        params.num_hooks = 1;
        params.source_parameters = NULL;
        params.num_source_parameters = 0;
        params.info_callback = NULL;
        params.info_priv = NULL;
        REQUIRE(pl_render_image(p->oracle, &source, &target, &params));
        pl_shader sh = pl_dispatch_begin(p->dp);
        REQUIRE(pl_shader_sample_direct(sh, pl_sample_src(.tex = views->frames[i].texture)));
        REQUIRE(pl_dispatch_finish(p->dp, pl_dispatch_params(
            .shader = &sh, .target = p->readback)));
        float actual[SH_N], expected[SH_N];
        REQUIRE(pl_tex_download(p->gpu, pl_tex_transfer_params(
            .tex = p->readback, .ptr = actual)));
        REQUIRE(pl_tex_download(p->gpu, pl_tex_transfer_params(
            .tex = p->expected, .ptr = expected)));
        float maximum = 0;
        for (int c = 0; c < SH_N; c++) {
            REQUIRE(isfinite(actual[c]) && isfinite(expected[c]));
            maximum = fmaxf(maximum, fabsf(actual[c] - expected[c]));
        }
        printf("- source hook oracle source=%d max=%.9g\n", p->base + i, maximum);
        REQUIRE(maximum < .003f);
        if (i == 1) {
            // Prove this oracle detects using entry-current PTS for source B/C.
            source_hook_parameter(p->oracle_hook, "PTS", p->owners[p->base].pts);
            REQUIRE(pl_render_image(p->oracle, &source, &target, &params));
            float wrong[SH_N], difference = 0;
            REQUIRE(pl_tex_download(p->gpu, pl_tex_transfer_params(
                .tex = p->expected, .ptr = wrong)));
            for (int c = 0; c < SH_N; c++) {
                REQUIRE(isfinite(wrong[c]));
                difference = fmaxf(difference, fabsf(expected[c] - wrong[c]));
            }
            printf("wrong-PTS precondition difference=%.9g\n", difference);
            REQUIRE(difference > .003f);
        }
    }
    return views->frames[views->reference].texture;
}

static void source_hook_tests(pl_gpu gpu, pl_fmt fmt)
{
    static const char shader[] =
        "//!PARAM gain\n//!TYPE DYNAMIC float\n"
        "//!MINIMUM 0\n//!MAXIMUM 1\n1\n"
        "//!PARAM PTS\n//!TYPE DYNAMIC float\n"
        "//!MINIMUM 0\n//!MAXIMUM 1\n0\n"
        "//!HOOK MAIN\n//!BIND HOOKED\n"
        "vec4 hook() {\n"
        "    vec4 color = HOOKED_texOff(0);\n"
        "    color.rgb *= gain + PTS * 0.5;\n"
        "    return color;\n"
        "}\n";
    struct source_hook_test p = {.gpu = gpu, .gain = 1};
    p.rr = pl_renderer_create(gpu->log, gpu);
    p.oracle = pl_renderer_create(gpu->log, gpu);
    p.dp = pl_dispatch_create(gpu->log, gpu);
    p.hook = pl_mpv_user_shader_parse(gpu, shader, strlen(shader));
    p.oracle_hook = pl_mpv_user_shader_parse(gpu, shader, strlen(shader));
    REQUIRE(p.rr && p.oracle && p.dp && p.hook && p.oracle_hook);
    REQUIRE(p.hook->num_parameters == 2);
    float pixels[SH_N];
    for (int i = 0; i < SH_W*SH_H; i++) {
        pixels[i*4] = .1f + (float)(i % 7) * .005f;
        pixels[i*4+1] = .15f;
        pixels[i*4+2] = .2f;
        pixels[i*4+3] = 1;
    }
    pl_tex tex[6] = {0};
    for (int i = 0; i < 6; i++) {
        tex[i] = pl_tex_create(gpu, pl_tex_params(
            .w = SH_W, .h = SH_H, .format = fmt, .sampleable = true,
            .renderable = true, .host_readable = true,
            .initial_data = i < 3 ? pixels : NULL));
        REQUIRE(tex[i]);
    }
    p.readback = tex[4];
    p.expected = tex[5];
    for (int i = 0; i < 3; i++) {
        p.owners[i] = (struct source_hook_owner){.p = &p, .pts = .1f * (float)(i+1)};
        p.sources[i] = source_hook_frame(tex[i]);
        p.sources[i].acquire = source_hook_acquire;
        p.sources[i].release = source_hook_release;
        p.sources[i].user_data = &p.owners[i];
    }
    struct pl_frame target = source_hook_frame(tex[3]);
    p.params = pl_render_default_params;
    p.params.peak_detect_params = NULL;
    p.params.frame_mixer = NULL;
    p.params.dither_params = NULL;
    const char output_shader[] =
        "//!PARAM output_gain\n//!TYPE DYNAMIC float\n"
        "//!MINIMUM 0\n//!MAXIMUM 1\n1\n"
        "//!HOOK OUTPUT\n//!BIND HOOKED\n"
        "vec4 hook() { return HOOKED_texOff(0) * output_gain; }\n";
    const struct pl_hook *output_hook = pl_mpv_user_shader_parse(
        gpu, output_shader, sizeof(output_shader) - 1);
    REQUIRE(output_hook && output_hook->num_parameters == 1);
    const struct pl_hook *hooks[] = {p.hook, output_hook};
    p.params.hooks = hooks;
    p.params.num_hooks = 2;
    const struct pl_hook_par *pts_par = NULL;
    for (int i = 0; i < p.hook->num_parameters; i++) {
        if (!strcmp(p.hook->parameters[i].name, "PTS"))
            pts_par = &p.hook->parameters[i];
    }
    REQUIRE(pts_par);
    const struct pl_hook_par *bindings[] = {pts_par, &output_hook->parameters[0]};
    const struct pl_hook_par *duplicates[] = {
        &output_hook->parameters[0], pts_par, pts_par,
    };
    p.params.source_parameters = bindings;
    p.params.num_source_parameters = PL_ARRAY_SIZE(bindings);
    p.params.info_callback = source_hook_info;
    p.params.info_priv = &p;
    struct pl_custom_frame_mixer mixer = {.priv = &p, .mix = source_hook_mix};
    uint64_t overlap = 0, previous[2] = {0};
    for (int step = 0; step < 12; step++) {
        p.base = step < 3 ? 0 : 1;
        p.alpha = .25f * (float)((step % 3) + 1);
        if (step >= 6) {
            p.alpha = .5f;
            p.gain = .5f;
        }
        source_hook_parameter(p.hook, "PTS", p.owners[p.base].pts);
        source_hook_parameter(p.hook, "gain", p.gain);
        const struct pl_frame *sources[] = {&p.sources[p.base], &p.sources[p.base+1]};
        uint64_t signatures[] = {(uint64_t)p.base+1, (uint64_t)p.base+2};
        double timestamps[] = {-p.alpha, 1.0-p.alpha};
        struct pl_frame_mix input = {.num_frames = 2, .frames = sources,
            .signatures = signatures, .timestamps = timestamps, .vsync_duration = .2f};
        if (step == 8) {
            p.params.source_parameters = duplicates;
            p.params.num_source_parameters = PL_ARRAY_SIZE(duplicates);
        } else if (step == 9) {
            // OUTPUT-only membership/value changes do not affect prepared views.
            p.params.source_parameters = bindings;
            p.params.num_source_parameters = 1;
            source_hook_parameter(output_hook, "output_gain", .5f);
        } else if (step == 10) {
            p.params.source_parameters = NULL;
            p.params.num_source_parameters = 0;
        } else if (step == 11) {
            p.params.source_parameters = bindings;
            p.params.num_source_parameters = PL_ARRAY_SIZE(bindings);
        }
        int before = p.frame0;
        REQUIRE(pl_render_image_mix_custom(p.rr, &input, &target, &p.params, &mixer));
        printf("- source hook step=%d prepared=%d\n", step, p.frame0 - before);
        int expected_prepares = step == 0 || step == 6 || step >= 10 ? 2 :
                                step == 3 ? 1 : 0;
        REQUIRE(p.frame0 - before == expected_prepares);
        if (step == 1 || step == 2 || step == 4 || step == 5 ||
            step == 7 || step == 8 || step == 9)
            REQUIRE(previous[0] == p.ids[0] && previous[1] == p.ids[1]);
        if (step == 2)
            overlap = p.ids[1];
        if (step == 3)
            REQUIRE(overlap == p.ids[0]);
        if (step == 6 || step >= 10)
            REQUIRE(previous[0] != p.ids[0] && previous[1] != p.ids[1]);
        memcpy(previous, p.ids, sizeof(previous));
    }
    REQUIRE(p.calls == 12);
    // Malformed lists must fail before any native acquire or mixer callback.
    struct pl_hook_par unrelated = *pts_par;
    const struct pl_hook_par *unknown[] = {&unrelated};
    const struct pl_hook_par *null_entry[] = {NULL};
    const struct pl_frame *frames[] = {&p.sources[1], &p.sources[2]};
    uint64_t signatures[] = {2, 3};
    double timestamps[] = {-.5, .5};
    struct pl_frame_mix input = {.num_frames = 2, .frames = frames,
        .signatures = signatures, .timestamps = timestamps, .vsync_duration = .2f};
    int acquired = 0;
    for (int i = 0; i < 3; i++)
        acquired += p.owners[i].acquired;
    for (int invalid = 0; invalid < 4; invalid++) {
        p.params.source_parameters = invalid == 0 ? bindings :
            invalid == 1 ? NULL : invalid == 2 ? unknown : null_entry;
        p.params.num_source_parameters = invalid == 0 ? -1 : 1;
        REQUIRE(!pl_render_image_mix_custom(p.rr, &input, &target, &p.params, &mixer));
        REQUIRE(p.calls == 12);
        int after = 0;
        for (int i = 0; i < 3; i++)
            after += p.owners[i].acquired;
        REQUIRE(after == acquired);
    }
    for (int i = 0; i < 3; i++) {
        REQUIRE(p.owners[i].acquired == p.owners[i].released);
        printf("- source hook source=%d acquire=%d release=%d\n", i,
               p.owners[i].acquired, p.owners[i].released);
    }
    pl_renderer_destroy(&p.rr);
    pl_renderer_destroy(&p.oracle);
    pl_dispatch_destroy(&p.dp);
    pl_mpv_user_shader_destroy(&output_hook);
    pl_mpv_user_shader_destroy(&p.hook);
    pl_mpv_user_shader_destroy(&p.oracle_hook);
    for (int i = 0; i < 6; i++)
        pl_tex_destroy(gpu, &tex[i]);
}

#ifdef PL_HAVE_DOVI
static void custom_mix_dovi_tests(pl_gpu gpu, pl_tex src, pl_tex dst)
{
    struct dovi_ref ref;
    struct pl_dovi_metadata metadata;
    pl_matrix3x3 matrix;
    dovi_test_ref(8, &ref);
    dovi_test_metadata(&ref, &metadata, &matrix);

    struct pl_frame image = custom_mix_frame(src);
    image.repr.sys = PL_COLOR_SYSTEM_DOLBYVISION;
    image.repr.dovi = &metadata;
    image.repr.bits = (struct pl_bit_encoding) {
        .sample_depth = 8, .color_depth = 8,
    };
    image.color = pl_color_space_hdr10;
    struct pl_frame el = image;
    el.repr.sys = PL_COLOR_SYSTEM_BT_2020_NC;
    el.repr.dovi = NULL;
    struct pl_frame target = custom_mix_frame(dst);
    target.color = pl_color_space_hdr10;

    const struct pl_frame *frames[] = {&image};
    const double timestamps[] = {0};
    uint64_t signature = 1;
    struct pl_frame_mix input = {
        .num_frames = 1, .frames = frames, .timestamps = timestamps,
        .signatures = &signature, .vsync_duration = 1,
    };
    struct custom_mix_test test = {.count = 1, .timestamps = timestamps};
    struct pl_custom_frame_mixer mixer = {.priv = &test, .mix = custom_mix_callback};
    struct pl_render_params params = pl_render_default_params;
    params.peak_detect_params = NULL;
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    REQUIRE(rr);

    // Ordinary rendering can display the base layer alone.
    REQUIRE(pl_render_image(rr, &image, &target, &params));
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE_CMP(test.calls, ==, 0, "d");

    image.enhancement_layer = &el;
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE_CMP(test.calls, ==, 1, "d");

    signature++;
    el.field = PL_FIELD_TOP;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE_CMP(test.calls, ==, 1, "d");

    signature++;
    el.field = PL_FIELD_NONE;
    image.field = image.first_field = PL_FIELD_TOP;
    params.deinterlace_params = &pl_deinterlace_default_params;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE_CMP(test.calls, ==, 1, "d");

    signature++;
    image.field = image.first_field = PL_FIELD_NONE;
    params.deinterlace_params = NULL;
    image.enhancement_layer = NULL;
    metadata.nlq_active = false;
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE_CMP(test.calls, ==, 2, "d");
    pl_renderer_destroy(&rr);
}
#endif

static void custom_mix_field_tests(pl_gpu gpu, pl_fmt fmt)
{
    enum { W = 16, H = 12, N = W * H * 4 };
    float pixels[N], actual[N], expected[N];
    for (int i = 0; i < W * H; i++) {
        float level = i / W % 2 ? 0.65f : 0.2f;
        for (int c = 0; c < 3; c++)
            pixels[4*i+c] = level;
        pixels[4*i+3] = 1.0f;
    }
    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .sampleable = true,
        .initial_data = pixels));
    pl_tex dst = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .renderable = true,
        .host_readable = true));
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    pl_renderer oracle = pl_renderer_create(gpu->log, gpu);
    pl_source_peak_state state = pl_source_peak_state_create(gpu->log, gpu);
    REQUIRE(src && dst && rr && oracle && state);
    struct pl_frame target = custom_mix_frame(dst);
    struct pl_frame fields[] = {custom_mix_frame(src), custom_mix_frame(src)};
    const struct pl_frame *frames[] = {&fields[0], &fields[1]};
    uint64_t signatures[] = {1, 2};
    double pts[] = {-0.25, 0.75};
    struct pl_frame_mix input = {
        .num_frames = 2, .frames = frames, .signatures = signatures,
        .timestamps = pts, .vsync_duration = 0.25,
    };
    struct custom_mix_test test = {.count = 2, .timestamps = pts};
    struct pl_custom_frame_mixer mixer = {.priv = &test, .mix = custom_mix_callback};
    struct pl_render_params params = pl_render_default_params;
    params.peak_detect_params = NULL;
    params.dither_params = NULL;
    const struct pl_deinterlace_params bob = {.algo = PL_DEINTERLACE_BOB};
    const struct pl_deinterlace_params weave = {.algo = PL_DEINTERLACE_WEAVE};
    uint64_t ids[] = {1, 1};
    struct pl_source_peak_context context = {.state = state, .ids = ids, .current_id = 1};

    for (int order = 0; order < 2; order++) {
        enum pl_field first = order == 0 ? PL_FIELD_TOP : PL_FIELD_BOTTOM;
        fields[0].field = fields[0].first_field = fields[1].first_field = first;
        fields[1].field = pl_field_other(first);
        context.epoch = order + 1;
        signatures[0] = 2 * order + 1;
        signatures[1] = 2 * order + 2;
        params.deinterlace_params = &bob;
        params.peak_detect_params = NULL;
        REQUIRE(pl_source_peak_submit(rr, state, context.epoch, 1, &fields[0],
                                      &pl_peak_detect_default_params) == PL_SOURCE_PEAK_READY);
        for (int ref = 0; ref < 2; ref++) {
            params.peak_detect_params = NULL;
            pts[0] = ref ? -0.75 : -0.25;
            pts[1] = pts[0] + 1.0;
            test.reference = ref;
            REQUIRE(pl_render_image(oracle, frames[ref], &target, &params));
            custom_mix_download(gpu, dst, expected);
            float wanted = fields[ref].field == PL_FIELD_TOP ? 0.2f : 0.65f;
            REQUIRE(fabsf(expected[4 * W] - wanted) < 0.003f);
            REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
            custom_mix_download(gpu, dst, actual);
            custom_mix_compare(actual, expected, N);
            REQUIRE(test.ids[0] != test.ids[1]);
            params.peak_detect_params = &pl_peak_detect_default_params;
            REQUIRE(pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
            custom_mix_download(gpu, dst, actual);
            custom_mix_compare(actual, expected, N);
        }
        int calls = test.calls;
        fields[1].field = first;
        REQUIRE(!pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
        fields[1].field = pl_field_other(first);
        params.deinterlace_params = &weave;
        REQUIRE(!pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
        params.deinterlace_params = NULL;
        params.peak_detect_params = NULL;
        REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        REQUIRE(test.calls == calls);
    }

    struct custom_mix_owner neighbors[2] = {0};
    struct pl_frame refs[] = {custom_mix_frame(src), custom_mix_frame(src)};
    for (int i = 0; i < 2; i++) {
        refs[i].acquire = custom_mix_acquire;
        refs[i].release = custom_mix_release;
        refs[i].user_data = &neighbors[i];
    }
    fields[0].prev = &refs[0];
    fields[1].next = &refs[1];
    signatures[0] = 5;
    signatures[1] = 6;
    params.deinterlace_params = &pl_deinterlace_default_params;
    params.peak_detect_params = &pl_peak_detect_default_params;
    REQUIRE(pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
    uint64_t cached[] = {test.ids[0], test.ids[1]};
    for (int i = 0; i < 2; i++)
        REQUIRE(neighbors[i].acquired == 1 && neighbors[i].released == 1);

    fields[1].next = &refs[0];
    signatures[1] = 7;
    REQUIRE(pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
    REQUIRE(test.ids[0] == cached[0] && test.ids[1] != cached[1]);
    REQUIRE(neighbors[0].acquired == 2 && neighbors[0].released == 2);
    REQUIRE(neighbors[1].acquired == 1 && neighbors[1].released == 1);

    // Cached output must not reacquire references supplied by acquire.
    struct custom_mix_owner output_owner = {.next = &refs[0]};
    fields[1].acquire = custom_mix_acquire;
    fields[1].release = custom_mix_release;
    fields[1].user_data = &output_owner;
    neighbors[0].fail = true;
    REQUIRE(pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
    REQUIRE(output_owner.acquired == 1 && output_owner.released == 1);
    REQUIRE(neighbors[0].acquired == 2 && neighbors[0].released == 2);
    neighbors[0].fail = false;

    pl_source_peak_state oracle_state = NULL;
    if (!gpu->glsl.compute || !gpu->limits.max_ssbo_size)
        goto done;
    oracle_state = pl_source_peak_state_create(gpu->log, gpu);
    REQUIRE(oracle_state);
    struct pl_peak_detect_params peak = pl_peak_detect_default_params;
    peak.allow_delayed = false;
    params.peak_detect_params = &peak;
    params.deinterlace_params = &bob;
    struct pl_frame native = custom_mix_frame(src);
    native.color = pl_color_space_hdr10;
    struct pl_source_peak_context oracle_context = {
        .state = oracle_state, .ids = ids, .current_id = 1,
    };
    struct custom_mix_owner pq_owner = {0};
    struct pl_frame woven = native;
    woven.acquire = custom_mix_acquire;
    woven.release = custom_mix_release;
    woven.user_data = &pq_owner;
    woven.prev = &refs[0];
    woven.next = &refs[1];

    for (int order = 0; order < 2; order++) {
        enum pl_field first = order == 0 ? PL_FIELD_TOP : PL_FIELD_BOTTOM;
        context.epoch = oracle_context.epoch = 3 + order;
        woven.field = woven.first_field = first;
        fields[0] = fields[1] = native;
        fields[0].field = fields[0].first_field = fields[1].first_field = first;
        fields[1].field = pl_field_other(first);
        signatures[0] = 8 + 2 * order;
        signatures[1] = 9 + 2 * order;
        enum pl_source_peak_status status = pl_source_peak_submit(
            rr, state, context.epoch, 1, &woven, &peak);
        REQUIRE(status == PL_SOURCE_PEAK_PENDING);
        struct pl_frame second = woven;
        second.field = pl_field_other(first);
        REQUIRE(pl_source_peak_submit(rr, state, context.epoch, 1, &second, &peak) == status);
        REQUIRE(pq_owner.acquired == order + 1 && pq_owner.released == order + 1);
        REQUIRE(neighbors[0].acquired == 2 && neighbors[1].acquired == 1);
        REQUIRE(pl_source_peak_submit(oracle, oracle_state, oracle_context.epoch,
                                     1, &native, &peak) == PL_SOURCE_PEAK_PENDING);

        for (int ref = 0; ref < 2; ref++) {
            pts[0] = ref ? -0.75 : -0.25;
            pts[1] = pts[0] + 1.0;
            test.reference = ref;
            REQUIRE(pl_render_image_source(oracle, frames[ref], &target, &params,
                                           &oracle_context));
            custom_mix_download(gpu, dst, expected);
            REQUIRE(pl_render_image_mix_source(rr, &input, &target, &params,
                                                &context, &mixer));
            custom_mix_download(gpu, dst, actual);
            custom_mix_compare(actual, expected, N);
            REQUIRE(test.ids[0] != test.ids[1]);
            struct pl_hdr_metadata measured = {0}, reference = {0};
            REQUIRE(pl_source_peak_get_hdr_metadata(state, context.epoch, 1, &measured));
            REQUIRE(pl_source_peak_get_hdr_metadata(oracle_state, oracle_context.epoch,
                                                    1, &reference));
            REQUIRE(measured.max_pq_y == reference.max_pq_y);
            REQUIRE(measured.avg_pq_y == reference.avg_pq_y);
            REQUIRE(measured.max_pq_y > 0.5f && measured.avg_pq_y < measured.max_pq_y);
            REQUIRE(pq_owner.acquired == order + 1 && pq_owner.released == order + 1);
        }
    }
    puts("PQ field mixes share the native woven source peak");

    context.epoch++;
    woven.color.transfer = PL_COLOR_TRC_HLG;
    fields[0].color = fields[1].color = woven.color;
    signatures[0] = 12;
    signatures[1] = 13;
    REQUIRE(pl_source_peak_submit(rr, state, context.epoch, 1, &woven, &peak) ==
            PL_SOURCE_PEAK_UNSUPPORTED);
    int calls = test.calls;
    REQUIRE(!pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
    REQUIRE(test.calls == calls);
    target.color = woven.color;
    struct pl_render_params hlg_oracle = params;
    hlg_oracle.peak_detect_params = NULL;
    REQUIRE(pl_render_image(oracle, frames[test.reference], &target, &hlg_oracle));
    custom_mix_download(gpu, dst, expected);
    REQUIRE(pl_render_image_mix_source(rr, &input, &target, &params, &context, &mixer));
    REQUIRE(test.calls == calls + 1);
    custom_mix_download(gpu, dst, actual);
    custom_mix_compare(actual, expected, N);
    struct pl_hdr_metadata metadata = {0};
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, context.epoch, 1, &metadata));

done:
    pl_renderer_destroy(&rr);
    pl_renderer_destroy(&oracle);
    pl_source_peak_state_destroy(&oracle_state);
    pl_source_peak_state_destroy(&state);
    pl_tex_destroy(gpu, &dst);
    pl_tex_destroy(gpu, &src);
}

static void pl_custom_mix_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fmt || !(fmt->caps & PL_FMT_CAP_BLITTABLE)) {
        printf("pl_custom_mix_tests: skipped (FP32 render/sample/read/clear required)\n");
        return;
    }
    source_hook_tests(gpu, fmt);
    custom_mix_field_tests(gpu, fmt);
    printf("pl_custom_mix_tests:\n");
    enum { W = 16, H = 12, N = W * H * 4 };
    float pixels[N], actual[N], expected[N];
    for (int i = 0; i < W * H; i++) {
        pixels[4*i] = 0.2f + 0.01f * (i % W);
        pixels[4*i+1] = 0.3f;
        pixels[4*i+2] = 0.4f;
        pixels[4*i+3] = 0.75f;
    }
    pl_tex src = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .sampleable = true,
        .host_writable = true, .initial_data = pixels));
    pl_tex dst = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .renderable = true,
        .blit_dst = fmt->caps & PL_FMT_CAP_BLITTABLE,
        .host_readable = true));
    REQUIRE(src && dst);
#ifdef PL_HAVE_DOVI
    custom_mix_dovi_tests(gpu, src, dst);
#endif
    struct pl_frame image = custom_mix_frame(src), target = custom_mix_frame(dst);
    struct pl_frame sources[16];
    const struct pl_frame *frames[16];
    uint64_t signatures[16];
    double timestamps[16];
    for (int i = 0; i < 16; i++) {
        sources[i] = image;
        frames[i] = &sources[i];
        signatures[i] = 100 + i;
        // Non-float-representable values, well outside normal kernel radius.
        timestamps[i] = i ? 32.0 * i + 0x1p-32 : -0.25 - 0x1p-40;
    }
    struct pl_frame_mix input = {
        .frames = frames, .signatures = signatures, .timestamps = timestamps,
        .vsync_duration = 1,
    };
    struct custom_mix_test test = {
        .gpu = gpu, .dispatch = pl_dispatch_create(gpu->log, gpu),
        .timestamps = timestamps,
    };
    struct pl_custom_frame_mixer mixer = {.priv = &test, .mix = custom_mix_callback};
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    pl_renderer oracle = pl_renderer_create(gpu->log, gpu);
    REQUIRE(rr && oracle && test.dispatch);
    struct pl_render_params params = pl_render_default_params;
    params.peak_detect_params = NULL;
    params.dither_params = NULL;
    params.background = PL_CLEAR_SKIP;
    params.frame_mixer = &pl_filter_bilinear;
    params.preserve_mixing_cache = true; // custom must still enforce view identity
    // Built-in interpolation can fail without disabling the explicit mixer.
    double next_timestamp = timestamps[1];
    timestamps[1] = 0.75;
    input.num_frames = test.count = 2;
    struct pl_frame_interp_params interp = pl_frame_interp_default_params;
    interp.motion.block_size = 0;
    params.frame_interp_params = &interp;
    REQUIRE(pl_render_image_mix(rr, &input, &target, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_FRAME_INTERPOLATION);
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(pl_renderer_get_errors(rr).errors == PL_RENDER_ERR_FRAME_INTERPOLATION);
    pl_renderer_reset(rr);
    params.frame_interp_params = NULL;
    timestamps[1] = next_timestamp;

    for (int n = 1; n <= 16; n++) {
        input.num_frames = test.count = n;
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    }
    uint64_t ids[16];
    memcpy(ids, test.ids, sizeof(ids));
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE_MEMEQ(ids, test.ids, sizeof(ids));
    timestamps[0] -= 0x1p-30;
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE_MEMEQ(ids, test.ids, sizeof(ids));
    REQUIRE(pl_render_image(oracle, &image, &target, &params));
    custom_mix_download(gpu, dst, expected);
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    custom_mix_download(gpu, dst, actual);
    custom_mix_compare(actual, expected, N);

    // A mutable parameter must invalidate views without changing hook pointers.
    static const char gain_shader[] =
        "//!PARAM gain\n"
        "//!TYPE DYNAMIC float\n"
        "//!MINIMUM 0.0\n"
        "//!MAXIMUM 1.0\n"
        "1.0\n"
        "//!HOOK MAIN\n"
        "//!BIND HOOKED\n"
        "vec4 hook() {\n"
        "    vec4 color = HOOKED_texOff(0);\n"
        "    color.rgb *= gain;\n"
        "    return color;\n"
        "}\n";
    const struct pl_hook *gain_hook = pl_mpv_user_shader_parse(
        gpu, gain_shader, sizeof(gain_shader) - 1);
    REQUIRE(gain_hook && gain_hook->num_parameters == 1);
    REQUIRE(gain_hook->parameters[0].type == PL_VAR_FLOAT);
    params.hooks = &gain_hook;
    params.num_hooks = 1;
    float previous[N];
    const float gains[] = {1.0f, 0.5f};
    for (int step = 0; step < PL_ARRAY_SIZE(gains); step++) {
        gain_hook->parameters[0].data->f = gains[step];
        REQUIRE(pl_render_image(oracle, &image, &target, &params));
        custom_mix_download(gpu, dst, expected);
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        custom_mix_download(gpu, dst, actual);
        printf("- custom hook gain step=%d value=%g\n", step, gains[step]);
        fflush(stdout);
        custom_mix_compare(actual, expected, N);
        if (step) {
            for (int i = 0; i < 16; i++)
                REQUIRE(ids[i] != test.ids[i]);
            double difference = 0;
            for (int i = 0; i < N; i++)
                difference = PL_MAX(difference, fabs(actual[i] - previous[i]));
            REQUIRE(difference > 0.003);
        }
        memcpy(previous, actual, sizeof(previous));
        memcpy(ids, test.ids, sizeof(ids));
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        REQUIRE_MEMEQ(ids, test.ids, sizeof(ids));
    }
    params.hooks = NULL;
    params.num_hooks = 0;
    pl_mpv_user_shader_destroy(&gain_hook);

    // Every view-changing operation must replace rendered-pixel identities.
    struct pl_color_adjustment adjustment = pl_color_adjustment_neutral;
    adjustment.brightness = 0.05f;
    for (int change = 0; change < 7; change++) {
        printf("- custom view identity change=%d\n", change);
        fflush(stdout);
        memcpy(ids, test.ids, sizeof(ids));
        switch (change) {
        case 0: target.color.transfer = PL_COLOR_TRC_BT_1886; break;
        case 1: target.color.hdr.min_luma = 0.02f; break;
        case 2: target.crop = (pl_rect2df){1, 1, W-1, H-1}; break;
        case 3: target.rotation = PL_ROTATION_180; break;
        case 4: params.color_adjustment = &adjustment; break;
        case 5: pl_renderer_flush_cache(rr); break;
        case 6: pl_renderer_reset(rr); break;
        }
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        for (int i = 0; i < 16; i++)
            REQUIRE(ids[i] != test.ids[i]);
    }
    target = custom_mix_frame(dst);
    params.color_adjustment = NULL;
    REQUIRE(pl_render_image_mix(rr, &input, &target, &params));
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(pl_render_image(oracle, &image, &target, &params));
    custom_mix_download(gpu, dst, expected);
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    custom_mix_download(gpu, dst, actual);
    custom_mix_compare(actual, expected, N);
    test.fail = true;
    int calls = test.calls;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(test.calls == calls + 1);
    custom_mix_download(gpu, dst, expected);
    REQUIRE_MEMEQ(actual, expected, sizeof(actual));
    test.fail = false;
    calls = test.calls;
    params.peak_detect_params = &pl_peak_detect_default_params;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    params.peak_detect_params = NULL;
    target.profile = (struct pl_icc_profile){.data = "x", .len = 1};
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    target.profile = (struct pl_icc_profile){0};
    target.repr.sys = PL_COLOR_SYSTEM_XYZ;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    target.repr.sys = PL_COLOR_SYSTEM_RGB;
    sources[0].field = PL_FIELD_TOP;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    sources[0].field = PL_FIELD_NONE;
    REQUIRE(test.calls == calls);
    struct pl_custom_lut *conversion = pl_lut_parse_cube(gpu->log, test_luts[0],
                                                        strlen(test_luts[0]));
    REQUIRE(conversion);
    params.lut = conversion;
    params.lut_type = PL_LUT_CONVERSION;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(test.calls == calls);
    // The restriction belongs to custom preparation, not ordinary rendering.
    REQUIRE(pl_render_image(oracle, &image, &target, &params));
    params.lut = NULL;
    params.lut_type = PL_LUT_UNKNOWN;
    pl_lut_free(&conversion);
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));

    // Overlay alpha makes both omission and double composition observable.
    for (int i = 0; i < W * H; i++) {
        pixels[4*i] = 0.8f; pixels[4*i+1] = 0.1f;
        pixels[4*i+2] = 0.2f; pixels[4*i+3] = 0.5f;
    }
    pl_tex overlay = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .sampleable = true,
        .initial_data = pixels));
    REQUIRE(overlay);
    struct pl_overlay_part parts[2] = {
        {.src = {0, 0, W, H}, .dst = {2, 2, 7, 6}},
        {.src = {0, 0, W, H}, .dst = {8, 5, 13, 10}},
    };
    struct pl_overlay overlays[2];
    for (int i = 0; i < 2; i++) {
        overlays[i] = (struct pl_overlay) {
            .tex = overlay, .mode = PL_OVERLAY_NORMAL,
            .repr = image.repr, .color = image.color,
            .num_parts = 1, .parts = &parts[i],
        };
    }
    input.num_frames = test.count = 1;
    sources[0].overlays = &overlays[0]; sources[0].num_overlays = 1;
    target.overlays = &overlays[1]; target.num_overlays = 1;
    for (int geometry = 0; geometry < 7; geometry++) {
        printf("- custom overlay geometry=%d\n", geometry);
        fflush(stdout);
        test.verify_clean = geometry == 0;
        test.encoded = target.color;
        if (geometry == 1) {
            sources[0].crop = (pl_rect2df){1, 1, W-1, H-1};
            sources[0].rotation = PL_ROTATION_180;
        } else if (geometry == 2) {
            target.crop = (pl_rect2df){W-1, 1, 1, H-1};
        } else if (geometry >= 3) {
            sources[0].rotation = geometry & 1 ? PL_ROTATION_90 : PL_ROTATION_270;
            target.rotation = geometry < 5 ? PL_ROTATION_90 : PL_ROTATION_270;
            sources[0].crop = (pl_rect2df){2, 1, W-1, H-2};
            target.crop = geometry & 1 ? (pl_rect2df){1, 2, W-2, H-1}
                                      : (pl_rect2df){W-2, H-1, 1, 2};
        }
        signatures[0]++;
        REQUIRE(pl_render_image(oracle, &sources[0], &target, &params));
        custom_mix_download(gpu, dst, expected);
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        custom_mix_download(gpu, dst, actual);
        custom_mix_compare(actual, expected, N);
    }
    test.verify_clean = false;
    // A different reference must supply the sole source overlay. Its overlay
    // descriptor is invalid outside acquire/release, including on cache hits.
    sources[0] = sources[1] = image;
    target = custom_mix_frame(dst);
    target.overlays = &overlays[1]; target.num_overlays = 1;
    struct custom_mix_owner source_owner = {
        .overlay = &overlays[0], .overlay_texture = overlay,
    };
    struct custom_mix_owner target_owner = {
        .set_crop = true, .crop = {1, 1, W-1, H-1},
    };
    sources[1].acquire = custom_mix_acquire;
    sources[1].release = custom_mix_release;
    sources[1].user_data = &source_owner;
    target.acquire = custom_mix_acquire;
    target.release = custom_mix_release;
    target.user_data = &target_owner;
    timestamps[0] = -32; timestamps[1] = 0.25 + 0x1p-40;
    input.num_frames = test.count = 2;
    test.reference = 1;
    signatures[0] += 30; signatures[1] += 30;
    for (int pass = 0; pass < 3; pass++) {
        if (pass == 2)
            target_owner.crop = (pl_rect2df){W-2, 1, 2, H-1};
        REQUIRE(pl_render_image(oracle, &sources[1], &target, &params));
        custom_mix_download(gpu, dst, expected);
        int source_calls = source_owner.acquired;
        int target_calls = target_owner.acquired;
        memcpy(ids, test.ids, sizeof(ids));
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        REQUIRE(source_owner.acquired == source_calls + (pass == 1 ? 1 : 2));
        REQUIRE(target_owner.acquired == target_calls + 1);
        REQUIRE(source_owner.acquired == source_owner.released);
        REQUIRE(target_owner.acquired == target_owner.released);
        REQUIRE(!source_owner.active && !target_owner.active && !overlays[0].tex);
        if (pass == 1) {
            REQUIRE(test.ids[0] == ids[0] && test.ids[1] == ids[1]);
        } else if (pass == 2) {
            REQUIRE(test.ids[0] != ids[0] && test.ids[1] != ids[1]);
        }
        custom_mix_download(gpu, dst, actual);
        custom_mix_compare(actual, expected, N);
    }
    // Fail while preparing the second source, after the target was acquired.
    // This must release both acquisitions once, without reaching the mixer.
    source_owner.fail = true;
    signatures[1]++;
    int source_calls = source_owner.acquired;
    int target_calls = target_owner.acquired;
    calls = test.calls;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(test.calls == calls);
    REQUIRE(source_owner.acquired == source_calls + 1);
    REQUIRE(target_owner.acquired == target_calls + 1);
    REQUIRE(source_owner.acquired == source_owner.released);
    REQUIRE(target_owner.acquired == target_owner.released);
    REQUIRE(!source_owner.active && !target_owner.active);
    source_owner.fail = false;
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(source_owner.acquired == source_owner.released);
    REQUIRE(target_owner.acquired == target_owner.released);
    // Populate legacy cache, then fail while replacing it with custom views.
    // Clearing the reported error must not make a partially replaced cache
    // look like a valid legacy result.
    REQUIRE(pl_render_image_mix(rr, &input, &target, &params));
    REQUIRE(pl_render_image_mix(oracle, &input, &target, &params));
    custom_mix_download(gpu, dst, expected);
    source_owner.fail = true;
    calls = test.calls;
    source_calls = source_owner.acquired;
    target_calls = target_owner.acquired;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(test.calls == calls);
    REQUIRE(source_owner.acquired == source_calls + 1);
    REQUIRE(target_owner.acquired == target_calls + 1);
    REQUIRE(source_owner.acquired == source_owner.released);
    REQUIRE(target_owner.acquired == target_owner.released);
    source_owner.fail = false;
    pl_renderer_reset_errors(rr, NULL);
    REQUIRE(pl_render_image_mix(rr, &input, &target, &params));
    custom_mix_download(gpu, dst, actual);
    custom_mix_compare(actual, expected, N);

    // Failed target acquisition is released once and never reaches a source
    // or the mixer. Releasing a borrowed inner target would double-release it.
    target_owner.fail = true;
    source_calls = source_owner.acquired;
    target_calls = target_owner.acquired;
    calls = test.calls;
    REQUIRE(!pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(test.calls == calls && source_owner.acquired == source_calls);
    REQUIRE(target_owner.acquired == target_calls + 1);
    REQUIRE(target_owner.acquired == target_owner.released);
    REQUIRE(!source_owner.active && !target_owner.active);
    target_owner.fail = false;
    REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
    REQUIRE(source_owner.acquired == source_owner.released);
    REQUIRE(target_owner.acquired == target_owner.released);
    test.reference = 0;
    timestamps[0] = -0.25 - 0x1p-40;
    timestamps[1] = 32 + 0x1p-32;
    input.num_frames = test.count = 1;
    pl_tex_destroy(gpu, &overlay);
    sources[0] = image;
    target = custom_mix_frame(dst);

    // Authored HDR adaptation remains per-source. Compare custom endpoints to
    // normal rendering, including nonzero display black and transparent RGB.
    const enum pl_color_transfer trcs[] = {PL_COLOR_TRC_GAMMA22,
                                           PL_COLOR_TRC_BT_1886};
    for (int t = 0; t < 2; t++) {
        target.color.transfer = trcs[t];
        for (int hdr = 0; hdr < 3; hdr++) {
            sources[0] = image;
            if (hdr) {
                sources[0].color = pl_color_space_hdr10;
                sources[0].color.transfer = hdr == 1 ? PL_COLOR_TRC_PQ
                                                    : PL_COLOR_TRC_HLG;
                sources[0].color.hdr.scene_max[0] = 600;
                sources[0].color.hdr.scene_max[1] = 500;
                sources[0].color.hdr.scene_max[2] = 400;
                sources[0].color.hdr.scene_avg = 80;
            }
            signatures[0]++;
            REQUIRE(pl_render_image(oracle, &sources[0], &target, &params));
            custom_mix_download(gpu, dst, expected);
            REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
            custom_mix_download(gpu, dst, actual);
            custom_mix_compare(actual, expected, N);
            if (hdr == 1) {
                // HDR source adaptation preserves the requested target peak;
                // SDR inference can instead replace it with the source peak.
                uint64_t old_id = test.ids[0];
                uint64_t source_signature = signatures[0];
                target.color.hdr.max_luma = 100;
                REQUIRE(pl_render_image(oracle, &sources[0], &target, &params));
                custom_mix_download(gpu, dst, expected);
                REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params,
                                                    &mixer));
                REQUIRE(signatures[0] == source_signature);
                REQUIRE(test.ids[0] != old_id);
                custom_mix_download(gpu, dst, actual);
                custom_mix_compare(actual, expected, N);
                printf("- custom HDR target peak identity trc=%d passed\n", trcs[t]);
                target.color.hdr.max_luma = 203;
            }
        }
    }

    // Linear mixing has a CPU oracle independent of renderer intermediate RGB.
    sources[0] = sources[1] = image;
    sources[0].color.transfer = sources[1].color.transfer = PL_COLOR_TRC_LINEAR_HDR;
    sources[0].color.hdr = sources[1].color.hdr = target.color.hdr;
    sources[1].color.hdr.max_luma = target.color.hdr.max_luma;
    float second_pixels[N];
    for (int i = 0; i < W * H; i++) {
        second_pixels[4*i] = 0.3f; second_pixels[4*i+1] = 0.2f;
        second_pixels[4*i+2] = 0.1f; second_pixels[4*i+3] = 0.25f;
    }
    pl_tex second = pl_tex_create(gpu, pl_tex_params(
        .w = W, .h = H, .format = fmt, .sampleable = true,
        .initial_data = second_pixels));
    REQUIRE(second);
    sources[1].planes[0].texture = second;
    signatures[0] += 100; signatures[1] += 100;
    input.num_frames = test.count = 2;
    test.blend = true;
    const float mix_phases[] = {0, 0.25f, 0.75f, 1};
    for (int phase = 0; phase < PL_ARRAY_SIZE(mix_phases); phase++) {
        test.alpha = mix_phases[phase];
        printf("- custom linear phase=%.9g\n", test.alpha);
        fflush(stdout);
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        custom_mix_download(gpu, dst, actual);
        // Original source texture was never changed by the overlay fixture.
        for (int i = 0; i < W * H; i++) {
            float a = (1-test.alpha)*0.75f + test.alpha*0.25f;
            float rgb[3] = {
                ((1-test.alpha)*(0.2f+0.01f*(i%W))*0.75f + test.alpha*0.3f*0.25f)/a,
                ((1-test.alpha)*0.3f*0.75f + test.alpha*0.2f*0.25f)/a,
                ((1-test.alpha)*0.4f*0.75f + test.alpha*0.1f*0.25f)/a,
            };
            pl_color_delinearize(&target.color, rgb);
            for (int c = 0; c < 3; c++) expected[4*i+c] = rgb[c];
            expected[4*i+3] = a;
        }
        custom_mix_compare(actual, expected, N);
    }
    test.blend = false;
    input.num_frames = test.count = 1;
    sources[0] = image;
    for (int opacity = 0; opacity <= 1; opacity++) {
        printf("- custom source opacity=%d\n", opacity);
        fflush(stdout);
        for (int i = 0; i < W * H; i++) {
            pixels[4*i] = 0.2f + 0.01f * (i % W);
            pixels[4*i+1] = 0.3f;
            pixels[4*i+2] = 0.4f;
            pixels[4*i+3] = opacity;
        }
        REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(.tex = src, .ptr = pixels)));
        signatures[0]++;
        REQUIRE(pl_render_image(oracle, &sources[0], &target, &params));
        custom_mix_download(gpu, dst, expected);
        REQUIRE(pl_render_image_mix_custom(rr, &input, &target, &params, &mixer));
        custom_mix_download(gpu, dst, actual);
        custom_mix_compare(actual, expected, N);
        for (int i = 0; i < W * H; i++)
            REQUIRE(fabsf(actual[4*i+3] - opacity) < 0.000001f);
    }
    pl_tex_destroy(gpu, &second);
    pl_tex_destroy(gpu, &test.output);
    pl_dispatch_destroy(&test.dispatch);
    pl_renderer_destroy(&oracle);
    pl_renderer_destroy(&rr);
    pl_tex_destroy(gpu, &src);
    pl_tex_destroy(gpu, &dst);
    printf("- custom mixer inputs, identities, overlays, HDR and linear phases passed\n");
}

static pl_shader peak_record_shader(pl_dispatch dispatch, pl_tex pattern,
                                     float level)
{
    pl_shader sh = pl_dispatch_begin(dispatch);
    REQUIRE(pl_shader_sample_direct(sh, pl_sample_src(.tex = pattern)));
    REQUIRE(pl_shader_custom(sh, &(struct pl_custom_shader) {
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .body = "color.rgb *= level;",
        .variables = &(struct pl_shader_var) {
            .var = pl_var_float("level"), .data = &level,
        },
        .num_variables = 1,
    }));
    return sh;
}

static void peak_record_metadata_equal(const struct pl_hdr_metadata *a,
                                       const struct pl_hdr_metadata *b)
{
    REQUIRE(a->max_pq_y == b->max_pq_y);
    REQUIRE(a->avg_pq_y == b->avg_pq_y);
    REQUIRE(a->scene_avg == b->scene_avg);
    for (int c = 0; c < 3; c++)
        REQUIRE(a->scene_max[c] == b->scene_max[c]);
}

static void peak_record_cpu_tests(void)
{
    // Analytic IIR and partial scene-change response. Values deliberately
    // exceed quantization epsilon; no production reduction helper is used.
    struct pl_peak_detect_params params = pl_peak_detect_default_params;
    params.smoothing_period = 2;
    for (int scene = 0; scene < 2; scene++) {
        params.scene_threshold_low = scene ? 1 : 0;
        params.scene_threshold_high = scene ? 10 : 0;
        struct sh_peak_result history = {
            .avg_pq = .2f, .max_pq = .2f, .scene_avg_pq = .2f,
            .scene_max_pq = .2f, .max_y_pq = .2f, .active_ratio = 1,
        };
        const struct sh_peak_result sample = {
            .avg_pq = .7f, .max_pq = .7f, .scene_avg_pq = .7f,
            .scene_max_pq = .7f, .max_y_pq = .7f, .active_ratio = .25f,
        };
        double expected = .2 + (1.0 - exp(-.5)) * (.7 - .2);
        if (scene) {
            double delta = .25 * (.7 - expected);
            double x = (delta - .01) / (.10 - .01);
            REQUIRE(x > 0 && x < 1); // exercise the hysteresis ramp, not an endpoint
            double weight = x*x*(3 - 2*x);
            expected += weight * (.7 - expected);
        }
        sh_peak_smooth(&history, &sample, &params);
        REQUIRE(fabs(history.avg_pq - expected) < 1e-6);
        REQUIRE(fabs(history.max_pq - expected) < 1e-6);
        REQUIRE(fabs(history.scene_avg_pq - expected) < 1e-6);
        REQUIRE(fabs(history.scene_max_pq - expected) < 1e-6);
        REQUIRE(fabs(history.max_y_pq - expected) < 1e-6);
        REQUIRE(history.active_ratio == .25f);
    }
}

static void peak_record_tests(pl_gpu gpu)
{
    peak_record_cpu_tests();
    struct sh_peak_result invalid = {.avg_pq = -1, .active_ratio = -1};
    struct sh_peak_result unchanged = invalid;
    REQUIRE(sh_get_peak_measurement(NULL, false, &invalid) == SH_PEAK_ERROR);
    REQUIRE_MEMEQ(&invalid, &unchanged, sizeof(invalid));
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fmt || !gpu->glsl.compute || !gpu->limits.max_ssbo_size)
        return;
    printf("peak_record_tests: measurement and temporal primitives\n");
    pl_dispatch dispatch = pl_dispatch_create(gpu->log, gpu);
    float pixels[32 * 32 * 4];
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            int i = 4 * (32 * y + x);
            for (int c = 0; c < 3; c++)
                pixels[i+c] = x < 16 && y < 16 ? 1 : 0;
            pixels[i+3] = 1;
        }
    }
    // Immutable across all queued measurements, including reverse readback.
    pl_tex pattern = pl_tex_create(gpu, pl_tex_params(
        .w = 32, .h = 32, .format = fmt, .sampleable = true,
        .initial_data = pixels));
    REQUIRE(pattern);
    pl_tex target = pl_tex_create(gpu, pl_tex_params(
        .w = 32, .h = 32, .format = fmt, .renderable = true,
        .host_readable = true));
    REQUIRE(dispatch && target);
    struct pl_color_space color = pl_color_space_hdr10;
    color.transfer = PL_COLOR_TRC_LINEAR_HDR;
    const float levels[] = {.1f, 4.0f, .5f};
    for (int delayed = 0; delayed < 2; delayed++) {
        struct pl_peak_detect_params params = pl_peak_detect_default_params;
        params.smoothing_period = 3;
        params.scene_threshold_low = 2;
        params.scene_threshold_high = 9;
        params.percentile = 93;
        params.black_cutoff = 2;
        params.allow_delayed = delayed;
        pl_shader_obj records[3] = {0}, legacy = NULL, selected = NULL;
        struct sh_peak_result raw[3], history = {0};
        for (int i = 0; i < 3; i++) {
            pl_shader sh = peak_record_shader(dispatch, pattern, levels[i]);
            REQUIRE(sh_measure_peak(sh, color, &records[i], &params));
            struct sh_peak_result sentinel = {
                .avg_pq = -1, .max_pq = -2, .scene_avg_pq = -3,
                .scene_max_pq = -4, .max_y_pq = -5, .active_ratio = -6,
            }, result = sentinel;
            REQUIRE(sh_get_peak_measurement(records[i], false, &result) ==
                    SH_PEAK_PENDING);
            REQUIRE_MEMEQ(&sentinel, &result, sizeof(result));
            REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
                .shader = &sh, .width = 32, .height = 32)));
        }
        // GPU completion/readback order must not be a temporal commit order.
        for (int i = 2; i >= 0; i--) {
            REQUIRE(sh_get_peak_measurement(records[i], true, &raw[i]) == SH_PEAK_READY);
            REQUIRE(raw[i].active_ratio == .25f);
            struct sh_peak_result again = {0};
            REQUIRE(sh_get_peak_measurement(records[i], false, &again) == SH_PEAK_READY);
            REQUIRE_MEMEQ(&raw[i], &again, sizeof(again));
        }
        for (int i = 0; i < 3; i++) {
            sh_peak_smooth(&history, &raw[i], &params);
            struct pl_hdr_metadata expected = {0}, actual = {0};
            REQUIRE(sh_peak_metadata(&history, &expected));
            pl_shader sh = peak_record_shader(dispatch, pattern, levels[i]);
            REQUIRE(pl_shader_detect_peak(sh, color, &legacy, &params));
            REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
                .shader = &sh, .width = 32, .height = 32)));
            pl_gpu_finish(gpu);
            REQUIRE(pl_get_detected_hdr_metadata(legacy, &actual));
            peak_record_metadata_equal(&actual, &expected);
            REQUIRE(sh_color_map_set_peak(gpu, &selected, &history));
            struct sh_peak_result saved = history;
            for (int repeat = 0; repeat < 3; repeat++) {
                REQUIRE(pl_get_detected_hdr_metadata(selected, &actual));
                peak_record_metadata_equal(&actual, &expected);
                REQUIRE(sh_get_peak_measurement(records[i], false, &raw[i]) == SH_PEAK_READY);
                REQUIRE_MEMEQ(&history, &saved, sizeof(history));
            }
            // Legacy smoothed state and selected snapshot must map identical pixels.
            float output[2][32 * 32 * 4];
            for (int path = 0; path < 2; path++) {
                sh = peak_record_shader(dispatch, pattern, levels[i]);
                pl_shader_color_map_ex(sh, &pl_color_map_default_params,
                    pl_color_map_args(.src = color, .dst = pl_color_space_bt709,
                                      .state = path ? &selected : &legacy));
                REQUIRE(pl_dispatch_finish(dispatch, pl_dispatch_params(
                    .shader = &sh, .target = target)));
                REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(
                    .tex = target, .ptr = output[path])));
            }
            for (int p = 0; p < 32 * 32 * 4; p++) {
                REQUIRE(isfinite(output[0][p]) && isfinite(output[1][p]));
                REQUIRE(fabsf(output[0][p] - output[1][p]) < 1e-6f);
            }
            REQUIRE(pl_get_detected_hdr_metadata(selected, &actual));
            peak_record_metadata_equal(&actual, &expected);
        }
        // A queued legacy shader cannot be replaced by a presentation snapshot.
        pl_shader sh = peak_record_shader(dispatch, pattern, levels[2]);
        struct sh_peak_result after_pending = history;
        sh_peak_smooth(&after_pending, &raw[2], &params);
        REQUIRE(pl_shader_detect_peak(sh, color, &legacy, &params));
        // Preflight failure must not replace/consume an already queued source,
        // even if the rejected invocation requests different measurement policy.
        struct pl_peak_detect_params rejected_params = params;
        rejected_params.black_cutoff += 1;
        pl_shader_obj original_state = legacy;
        pl_shader rejected = peak_record_shader(dispatch, pattern, 9);
        rejected->type = SH_FRAGMENT;
        REQUIRE(!pl_shader_detect_peak(rejected, color, &legacy, &rejected_params));
        REQUIRE(legacy == original_state);
        pl_dispatch_abort(dispatch, &rejected);
        pl_shader_obj fresh_measurement = NULL;
        rejected = peak_record_shader(dispatch, pattern, 9);
        rejected->type = SH_FRAGMENT;
        REQUIRE(!sh_measure_peak(rejected, color, &fresh_measurement, &params));
        REQUIRE(!fresh_measurement);
        pl_dispatch_abort(dispatch, &rejected);
        REQUIRE(!sh_color_map_set_peak(gpu, &legacy, &history));
        REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
            .shader = &sh, .width = 32, .height = 32)));
        pl_gpu_finish(gpu);
        struct pl_hdr_metadata measured = {0};
        REQUIRE(pl_get_detected_hdr_metadata(legacy, &measured));
        struct pl_hdr_metadata pending_expected = {0};
        REQUIRE(sh_peak_metadata(&after_pending, &pending_expected));
        peak_record_metadata_equal(&measured, &pending_expected);
        pl_reset_detected_peak(legacy);
        REQUIRE(sh_color_map_set_peak(gpu, &legacy, &history));
        struct pl_hdr_metadata selected_meta = {0}, expected = {0};
        REQUIRE(pl_get_detected_hdr_metadata(legacy, &selected_meta));
        REQUIRE(sh_peak_metadata(&history, &expected));
        peak_record_metadata_equal(&selected_meta, &expected);
        for (int i = 0; i < 3; i++) pl_shader_obj_destroy(&records[i]);
        pl_shader_obj_destroy(&legacy);
        pl_shader_obj_destroy(&selected);
    }

    // Changing only black_cutoff must reset legacy smoothing, as a fresh state.
    // Disable scene snapping here so it cannot hide a missing policy reset.
    struct pl_peak_detect_params params = pl_peak_detect_default_params;
    params.scene_threshold_low = 0;
    params.scene_threshold_high = 0;
    params.allow_delayed = false;
    params.black_cutoff = 0;
    printf("- peak cutoff-only reset without scene snapping\n");
    fflush(stdout);
    pl_shader_obj reused = NULL, fresh = NULL;
    pl_shader sh = peak_record_shader(dispatch, pattern, 4);
    REQUIRE(pl_shader_detect_peak(sh, color, &reused, &params));
    REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh, .width = 32, .height = 32)));
    pl_gpu_finish(gpu);
    struct pl_hdr_metadata before = {0};
    REQUIRE(pl_get_detected_hdr_metadata(reused, &before));
    params.black_cutoff = 40;
    struct pl_hdr_metadata results[2] = {0};
    for (int path = 0; path < 2; path++) {
        sh = peak_record_shader(dispatch, pattern, .0001f);
        REQUIRE(pl_shader_detect_peak(sh, color, path ? &fresh : &reused, &params));
        REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
            .shader = &sh, .width = 32, .height = 32)));
        pl_gpu_finish(gpu);
        REQUIRE(pl_get_detected_hdr_metadata(path ? fresh : reused, &results[path]));
    }
    peak_record_metadata_equal(&results[0], &results[1]);
    REQUIRE(fabsf(before.avg_pq_y - results[0].avg_pq_y) > .1f);
    pl_shader_obj_destroy(&reused);
    pl_shader_obj_destroy(&fresh);

    // Solid black has no active workgroups and remains a valid immutable result.
    pl_shader_obj black = NULL;
    sh = peak_record_shader(dispatch, pattern, 0);
    REQUIRE(sh_measure_peak(sh, color, &black, &params));
    REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh, .width = 32, .height = 32)));
    struct sh_peak_result result = {0};
    REQUIRE(sh_get_peak_measurement(black, true, &result) == SH_PEAK_READY);
    REQUIRE(result.active_ratio == 0);
    REQUIRE(result.avg_pq == PL_COLOR_HDR_BLACK && result.max_pq == PL_COLOR_HDR_BLACK);
    pl_shader_obj_destroy(&black);
    pl_dispatch_destroy(&dispatch);
    pl_tex_destroy(gpu, &target);
    pl_tex_destroy(gpu, &pattern);
    printf("- peak primitive ordering, immutable snapshots and cutoff reset passed\n");
}

static void source_time_expect(pl_source_peak_state state, uint64_t epoch,
                               uint64_t id, const struct sh_peak_result *result)
{
    struct pl_hdr_metadata actual = {0}, expected = {0};
    REQUIRE(pl_source_peak_get_hdr_metadata(state, epoch, id, &actual));
    REQUIRE(sh_peak_metadata(result, &expected));
    peak_record_metadata_equal(&actual, &expected);
}

struct source_time_info { int frames, blends; };

static void source_time_info_cb(void *priv, const struct pl_render_info *info)
{
    struct source_time_info *counts = priv;
    if (info->stage == PL_RENDER_STAGE_BLEND)
        counts->blends++;
    else
        counts->frames++;
}

struct source_time_reentry {
    pl_renderer rr;
    pl_source_peak_state state;
    const struct pl_frame *source;
    const struct pl_peak_detect_params *params;
    int acquired, released;
};

static void source_time_reenter(struct source_time_reentry *owner)
{
    REQUIRE(!pl_source_peak_state_reset(owner->state, 99));
    REQUIRE(pl_source_peak_submit(owner->rr, owner->state, 5, 200,
                                 owner->source, owner->params) == PL_SOURCE_PEAK_ERROR);
}

static bool source_time_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    (void) gpu;
    struct source_time_reentry *owner = frame->user_data;
    owner->acquired++;
    source_time_reenter(owner);
    return true;
}

static void source_time_release(pl_gpu gpu, struct pl_frame *frame)
{
    (void) gpu;
    struct source_time_reentry *owner = frame->user_data;
    owner->released++;
    source_time_reenter(owner);
}

static void source_time_pixels(pl_gpu gpu, pl_renderer rr,
                               pl_dispatch dispatch, pl_tex dst,
                               const struct pl_frame *dark,
                               const struct pl_frame *bright)
{
    // Include sub-peak values so changing the estimator is observable, rather
    // than comparing only black and fully clipped white endpoints.
    float data[32 * 32 * 4];
    const float levels[] = {.05f, .2f, .5f, 4};
    for (int i = 0; i < 32 * 32; i++) {
        for (int c = 0; c < 3; c++)
            data[4*i+c] = levels[(i % 32) / 8];
        data[4*i+3] = 1;
    }
    pl_tex tex = pl_tex_create(gpu, pl_tex_params(
        .w = 32, .h = 32, .format = dst->params.format,
        .sampleable = true, .initial_data = data));
    REQUIRE(tex);
    struct pl_frame source = *bright;
    source.planes[0].texture = tex;
    source.acquire = NULL;
    source.release = NULL;
    struct pl_frame target = custom_mix_frame(dst);
    pl_renderer oracle = pl_renderer_create(gpu->log, gpu);
    pl_source_peak_state state = pl_source_peak_state_create(gpu->log, gpu);
    REQUIRE(oracle && state);
    struct pl_peak_detect_params peak = pl_peak_detect_default_params;
    peak.smoothing_period = 3;
    peak.scene_threshold_low = peak.scene_threshold_high = 0;
    peak.allow_delayed = false;
    struct pl_color_map_params mapping = pl_color_map_default_params;
    mapping.mastering_clip = 0;
    struct pl_color_adjustment adjustment = pl_color_adjustment_neutral;
    adjustment.brightness = .03f;
    struct pl_render_params params = pl_render_default_params;
    params.peak_detect_params = &peak;
    params.color_map_params = &mapping;
    params.color_adjustment = &adjustment;
    params.dither_params = NULL;
    params.frame_mixer = NULL;
    params.skip_caching_single_frame = true;
    const struct pl_frame *frames[] = {&source};
    uint64_t id = 40, signature = 777;
    double timestamp = 0;
    struct pl_frame_mix mix = {
        .num_frames = 1, .frames = frames, .signatures = &signature,
        .timestamps = &timestamp, .vsync_duration = 1,
    };
    struct pl_source_peak_context context = {
        .state = state, .epoch = 1, .ids = &id, .current_id = id,
    };
    struct custom_mix_test custom = {
        .gpu = gpu, .dispatch = dispatch, .count = 1, .timestamps = &timestamp,
    };
    struct pl_custom_frame_mixer mixer = {.priv = &custom, .mix = custom_mix_callback};
    REQUIRE(pl_source_peak_submit(rr, state, 1, 10, dark, &peak) >= 0);
    REQUIRE(pl_source_peak_submit(rr, state, 1, id, &source, &peak) >= 0);
    float actual[32 * 32 * 4], expected[32 * 32 * 4], previous[32 * 32 * 4];
    uint64_t previous_id = 0;
    for (int change = 0; change < 3; change++) {
        if (change == 1) {
            peak.allow_delayed = true;
            REQUIRE(pl_source_peak_submit(rr, state, 1, id, &source, &peak) >= 0);
        } else if (change == 2) {
            REQUIRE(pl_source_peak_state_reset(state, 2));
            context.epoch = 2;
            peak.allow_delayed = false;
            REQUIRE(pl_source_peak_submit(rr, state, 2, id, &source, &peak) >= 0);
        }
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
        custom_mix_download(gpu, dst, actual);
        struct pl_frame reference = source;
        REQUIRE(pl_source_peak_get_hdr_metadata(state, context.epoch, id,
                                                &reference.color.hdr));
        struct pl_render_params reference_params = params;
        reference_params.peak_detect_params = NULL;
        REQUIRE(pl_render_image(oracle, &reference, &target, &reference_params));
        custom_mix_download(gpu, dst, expected);
        printf("source-time pixel change=%d\n", change);
        fflush(stdout);
        custom_mix_compare(actual, expected, 32 * 32 * 4);
        if (change) {
            float difference = 0;
            for (int i = 0; i < 32 * 32 * 4; i++)
                difference = PL_MAX(difference, fabsf(expected[i] - previous[i]));
            printf("source-time observable change=%d max=%g\n", change, difference);
            fflush(stdout);
            REQUIRE(difference > .003f);
        }
        memcpy(previous, expected, sizeof(previous));
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
        custom_mix_download(gpu, dst, actual);
        custom_mix_compare(actual, expected, 32 * 32 * 4);
        if (change)
            REQUIRE(custom.ids[0] != previous_id);
        previous_id = custom.ids[0];
        // Confirm the view adjustment is visible in the independent oracle.
        reference_params.color_adjustment = NULL;
        REQUIRE(pl_render_image(oracle, &reference, &target, &reference_params));
        custom_mix_download(gpu, dst, actual);
        float difference = 0;
        for (int i = 0; i < 32 * 32 * 4; i++)
            difference = PL_MAX(difference, fabsf(actual[i] - expected[i]));
        REQUIRE(difference > .003f);
    }
    pl_renderer_flush_cache(rr);
    pl_source_peak_state_destroy(&state);
    pl_renderer_destroy(&oracle);
    pl_tex_destroy(gpu, &tex);
}

static void source_time_image(pl_gpu gpu, pl_tex dst,
                              const struct pl_frame native[3],
                              const struct sh_peak_result after[3])
{
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    pl_renderer oracle = pl_renderer_create(gpu->log, gpu);
    pl_source_peak_state state = pl_source_peak_state_create(gpu->log, gpu);
    REQUIRE(rr && oracle && state);
    struct pl_peak_detect_params peak = pl_peak_detect_default_params;
    peak.smoothing_period = 3;
    peak.scene_threshold_low = peak.scene_threshold_high = 0;
    peak.percentile = 97;
    peak.black_cutoff = 2;
    peak.allow_delayed = false;
    struct pl_color_map_params mapping = pl_color_map_default_params;
    mapping.mastering_clip = 0;
    struct pl_render_params params = pl_render_default_params;
    params.peak_detect_params = &peak;
    params.color_map_params = &mapping;
    params.dither_params = NULL;
    struct pl_frame target = custom_mix_frame(dst);
    // Even an uninitialized state's empty render protects callback reentry.
    struct source_time_reentry reentry = {
        .rr = rr, .state = state, .source = &native[0], .params = &peak,
    };
    struct pl_frame guarded_target = target;
    guarded_target.acquire = source_time_acquire;
    guarded_target.release = source_time_release;
    guarded_target.user_data = &reentry;
    struct pl_source_peak_context uninitialized = {.state = state};
    REQUIRE(pl_render_image_source(rr, NULL, &guarded_target, &params, &uninitialized));
    REQUIRE(reentry.acquired == 1 && reentry.released == 1);
    const uint64_t ids[] = {10, 40, 90};
    for (int i = 0; i < 3; i++)
        REQUIRE(pl_source_peak_submit(rr, state, 1, ids[i], &native[i], &peak) >= 0);
    uint64_t view_id = 10;
    struct pl_source_peak_context context = {
        .state = state, .epoch = 1, .ids = &view_id, .current_id = 10,
    };
    float actual[32 * 32 * 4], expected[32 * 32 * 4], previous[32 * 32 * 4];
    // The native identity selects metadata, not a cache key for materialized pixels.
    for (int i = 0; i < 2; i++) {
        struct pl_frame view = native[i];
        view.acquire = NULL;
        view.release = NULL;
        REQUIRE(pl_render_image_source(rr, &view, &target, &params, &context));
        custom_mix_download(gpu, dst, actual);
        REQUIRE(pl_source_peak_get_hdr_metadata(state, 1, view_id, &view.color.hdr));
        struct pl_render_params reference = params;
        reference.peak_detect_params = NULL;
        REQUIRE(pl_render_image(oracle, &view, &target, &reference));
        custom_mix_download(gpu, dst, expected);
        custom_mix_compare(actual, expected, 32 * 32 * 4);
        if (i) {
            float difference = 0;
            for (int j = 0; j < 32 * 32 * 4; j++)
                difference = PL_MAX(difference, fabsf(actual[j] - previous[j]));
            REQUIRE(difference > .003f);
        }
        memcpy(previous, actual, sizeof(previous));
    }
    // Distinguish the view's snapshot from the public current-source snapshot.
    // Fixed HDR pixels include sub-peak values and highlights above the low
    // source peak. Prove both oracle outputs differ before testing the API.
    float materialized[32 * 32 * 4];
    const float levels[] = {.05f, .2f, .5f, 4};
    for (int i = 0; i < 32 * 32; i++) {
        for (int c = 0; c < 3; c++)
            materialized[4*i+c] = levels[(i % 32) / 8];
        materialized[4*i+3] = 1;
    }
    pl_tex materialized_tex = pl_tex_create(gpu, pl_tex_params(
        .w = 32, .h = 32, .format = dst->params.format,
        .sampleable = true, .initial_data = materialized));
    REQUIRE(materialized_tex);
    struct pl_frame fixed = native[1], dim_target = target;
    fixed.planes[0].texture = materialized_tex;
    fixed.acquire = NULL;
    fixed.release = NULL;
    dim_target.color.hdr.max_luma = 50;
    struct pl_render_params reference = params;
    reference.peak_detect_params = NULL;
    struct pl_frame view_reference = fixed, wrong_reference = fixed;
    REQUIRE(sh_peak_metadata(&after[0], &view_reference.color.hdr));
    REQUIRE(sh_peak_metadata(&after[1], &wrong_reference.color.hdr));
    REQUIRE(pl_render_image(oracle, &view_reference, &dim_target, &reference));
    custom_mix_download(gpu, dst, expected);
    REQUIRE(pl_render_image(oracle, &wrong_reference, &dim_target, &reference));
    custom_mix_download(gpu, dst, previous);
    float snapshot_difference = 0;
    for (int i = 0; i < 32 * 32 * 4; i++) {
        REQUIRE(isfinite(expected[i]) && isfinite(previous[i]));
        snapshot_difference = PL_MAX(snapshot_difference, fabsf(expected[i] - previous[i]));
    }
    printf("direct source-time view/current oracle difference=%g\n", snapshot_difference);
    fflush(stdout);
    REQUIRE(snapshot_difference > .003f);
    context.current_id = 40;
    REQUIRE(pl_render_image_source(rr, &fixed, &dim_target, &params, &context));
    custom_mix_download(gpu, dst, actual);
    custom_mix_compare(actual, expected, 32 * 32 * 4);
    float wrong_difference = 0;
    for (int i = 0; i < 32 * 32 * 4; i++)
        wrong_difference = PL_MAX(wrong_difference, fabsf(actual[i] - previous[i]));
    REQUIRE(wrong_difference > .003f);
    struct pl_hdr_metadata public_current = {0}, expected_current = {0};
    REQUIRE(pl_renderer_get_hdr_metadata(rr, &public_current));
    REQUIRE(sh_peak_metadata(&after[1], &expected_current));
    peak_record_metadata_equal(&public_current, &expected_current);
    pl_tex_destroy(gpu, &materialized_tex);

    view_id = 90;
    context.current_id = 40;
    REQUIRE(pl_render_image_source(rr, &native[2], &target, &params, &context));
    struct pl_hdr_metadata observed = {0}, current = {0};
    REQUIRE(pl_renderer_get_hdr_metadata(rr, &observed));
    REQUIRE(sh_peak_metadata(&after[1], &current));
    peak_record_metadata_equal(&observed, &current);
    source_time_expect(state, 1, 90, &after[2]);

    // Neither failed acquisition nor target rendering may advance or retire A.
    struct custom_mix_owner failure = {.fail = true};
    struct pl_frame failed = native[2];
    failed.user_data = &failure;
    context.current_id = 90;
    context.retire_before_id = 40;
    REQUIRE(!pl_render_image_source(rr, &failed, &target, &params, &context));
    REQUIRE(failure.acquired == 1 && failure.released == 1 && !failure.active);
    struct pl_frame failed_target = target;
    failed_target.acquire = custom_mix_acquire;
    failed_target.release = custom_mix_release;
    failed_target.user_data = &failure;
    REQUIRE(!pl_render_image_source(rr, &native[2], &failed_target, &params, &context));
    REQUIRE(failure.acquired == 2 && failure.released == 2 && !failure.active);
    REQUIRE(pl_renderer_get_hdr_metadata(rr, &observed));
    peak_record_metadata_equal(&observed, &current);
    source_time_expect(state, 1, 10, &after[0]);
    context.current_id = 40;
    context.retire_before_id = 0;
    view_id = 10;
    REQUIRE(pl_render_image_source(rr, &native[0], &target, &params, &context));

    // Empty views report no metadata and ignore clock/IDs without committing.
    struct pl_source_peak_context empty_context = {.state = state};
    REQUIRE(!pl_render_image_source(rr, NULL, &failed_target, &params, &empty_context));
    REQUIRE(pl_renderer_get_hdr_metadata(rr, &observed));
    peak_record_metadata_equal(&observed, &current);
    REQUIRE(pl_render_image_source(rr, NULL, &target, &params, &empty_context));
    REQUIRE(!pl_renderer_get_hdr_metadata(rr, &observed));
    REQUIRE(pl_render_image_source(rr, &native[0], &target, &params, &context));
    struct pl_frame_mix empty_mix = {0};
    REQUIRE(pl_render_image_mix_source(rr, &empty_mix, &target, &params,
                                       &empty_context, NULL));
    REQUIRE(!pl_renderer_get_hdr_metadata(rr, &observed));
    source_time_expect(state, 1, 10, &after[0]);

    view_id = 90;
    context.retire_before_id = 50; // exceeds current B
    REQUIRE(!pl_render_image_source(rr, &native[2], &target, &params, &context));
    context.current_id = 90;
    context.retire_before_id = 40;
    REQUIRE(pl_render_image_source(rr, &native[2], &target, &params, &context));
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, 10, &observed));
    context.current_id = 40;
    REQUIRE(!pl_render_image_source(rr, &native[2], &target, &params, &context));

    // An unsupported view uses ordinary rendering, but cannot leak legacy peaks.
    struct pl_frame hlg = native[1];
    hlg.color.transfer = PL_COLOR_TRC_HLG;
    hlg.acquire = NULL;
    hlg.release = NULL;
    REQUIRE(pl_source_peak_submit(rr, state, 1, 120, &hlg, &peak) ==
            PL_SOURCE_PEAK_UNSUPPORTED);
    view_id = context.current_id = 120;
    REQUIRE(pl_render_image_source(rr, &hlg, &target, &params, &context));
    custom_mix_download(gpu, dst, actual);
    pl_renderer_flush_cache(oracle);
    REQUIRE(pl_render_image(oracle, &hlg, &target, &params));
    custom_mix_download(gpu, dst, expected);
    custom_mix_compare(actual, expected, 32 * 32 * 4);
    REQUIRE(!pl_renderer_get_hdr_metadata(rr, &observed));
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, 120, &observed));
    REQUIRE(pl_render_image_source(rr, NULL, &target, &params, &empty_context));
    hlg.planes[0].texture = native[0].planes[0].texture;
    REQUIRE(pl_source_peak_submit(rr, state, 1, 140, &hlg, &peak) ==
            PL_SOURCE_PEAK_UNSUPPORTED);
    view_id = context.current_id = 140;
    REQUIRE(pl_render_image_source(rr, &hlg, &target, &params, &context));
    custom_mix_download(gpu, dst, actual);
    REQUIRE(pl_render_image(oracle, &hlg, &target, &params));
    custom_mix_download(gpu, dst, expected);
    custom_mix_compare(actual, expected, 32 * 32 * 4);
    REQUIRE(!pl_renderer_get_hdr_metadata(rr, &observed));
    pl_renderer_destroy(&rr);
    pl_renderer_destroy(&oracle);
    pl_source_peak_state_destroy(&state);
    printf("- direct source-time pixels, clock, failure, empty and boundary passed\n");
}

static void source_time_hlg_views(pl_gpu gpu, pl_dispatch dispatch, pl_tex dst)
{
    struct pl_frame hlg[2];
    pl_tex textures[2] = {0};
    float pixels[32 * 32 * 4];
    for (int n = 0; n < 2; n++) {
        for (int i = 0; i < 32 * 32; i++) {
            pixels[4*i+0] = (i % 32) / 50.0f + .1f * n;
            pixels[4*i+1] = .3f + .1f * n;
            pixels[4*i+2] = .5f - .1f * n;
            pixels[4*i+3] = 1;
        }
        textures[n] = pl_tex_create(gpu, pl_tex_params(
            .w = 32, .h = 32, .format = dst->params.format,
            .sampleable = true, .initial_data = pixels));
        REQUIRE(textures[n]);
        hlg[n] = custom_mix_frame(textures[n]);
        hlg[n].color = pl_color_space_bt2020_hlg;
    }

    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    pl_renderer oracle = pl_renderer_create(gpu->log, gpu);
    pl_source_peak_state state = pl_source_peak_state_create(gpu->log, gpu);
    REQUIRE(rr && oracle && state);
    struct pl_peak_detect_params peak = pl_peak_detect_default_params;
    peak.allow_delayed = false;
    struct pl_render_params params = pl_render_default_params;
    params.peak_detect_params = &peak;
    params.dither_params = NULL;
    const struct pl_frame *frames[] = {&hlg[0], &hlg[1]};
    const uint64_t ids[] = {10, 20}, signatures[] = {40, 50};
    const double times[] = {-.25, .75};
    struct pl_frame_mix mix = {
        .num_frames = 2, .frames = frames, .signatures = signatures,
        .timestamps = times, .vsync_duration = 1,
    };
    struct pl_source_peak_context context = {
        .state = state, .epoch = 1, .ids = ids, .current_id = ids[0],
    };
    struct custom_mix_test custom = {
        .gpu = gpu, .dispatch = dispatch, .count = 2, .timestamps = times,
        .blend = true, .alpha = .25f,
    };
    struct pl_custom_frame_mixer mixer = {.priv = &custom, .mix = custom_mix_callback};
    for (int n = 0; n < 2; n++)
        REQUIRE(pl_source_peak_submit(rr, state, 1, ids[n], frames[n], &peak) ==
                PL_SOURCE_PEAK_UNSUPPORTED);

    struct pl_frame target = custom_mix_frame(dst);
    REQUIRE(!pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    REQUIRE(custom.calls == 0);

    target.color = pl_color_space_hdr10;
    target.color.hdr.min_luma = .005f;
    float actual[32 * 32 * 4], previous = 0;
    uint64_t view_ids[2] = {0};
    const float peaks[] = {600, 1000};
    for (int n = 0; n < PL_ARRAY_SIZE(peaks); n++) {
        target.color.hdr.max_luma = peaks[n];
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
        REQUIRE(custom.calls == n + 1);
        custom_mix_download(gpu, dst, actual);
        for (int i = 0; i < 2; i++) {
            REQUIRE(custom.ids[i] != view_ids[i]);
            view_ids[i] = custom.ids[i];
        }
        // Ordinary endpoints provide an independent encoding and mapping
        // oracle. Decode their output before mixing their display light.
        float endpoints[2][32 * 32 * 4];
        for (int i = 0; i < 2; i++) {
            REQUIRE(pl_render_image(oracle, frames[i], &target, &params));
            custom_mix_download(gpu, dst, endpoints[i]);
        }
        float expected[32 * 32 * 4];
        for (int i = 0; i < 32 * 32; i++) {
            float a[3], b[3];
            memcpy(a, &endpoints[0][4*i], sizeof(a));
            memcpy(b, &endpoints[1][4*i], sizeof(b));
            pl_color_linearize(&target.color, a);
            pl_color_linearize(&target.color, b);
            for (int c = 0; c < 3; c++)
                a[c] += custom.alpha * (b[c] - a[c]);
            pl_color_delinearize(&target.color, a);
            memcpy(&expected[4*i], a, sizeof(a));
            expected[4*i+3] = 1;
        }
        custom_mix_compare(actual, expected, 32 * 32 * 4);
        if (n)
            REQUIRE(fabsf(actual[2] - previous) > .003f);
        previous = actual[2];
        struct pl_hdr_metadata observed;
        REQUIRE(!pl_renderer_get_hdr_metadata(rr, &observed));
    }

    target = custom_mix_frame(dst);
    REQUIRE(!pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    REQUIRE(custom.calls == 2);
    pl_tex_destroy(gpu, &custom.output);
    for (int i = 0; i < 2; i++)
        pl_tex_destroy(gpu, &textures[i]);
    pl_source_peak_state_destroy(&state);
    pl_renderer_destroy(&oracle);
    pl_renderer_destroy(&rr);
    printf("- target-relative HLG prepared views and SDR peak guard passed\n");
}

static void source_time_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE |
        PL_FMT_CAP_BLITTABLE);
    if (!fmt || !gpu->glsl.compute || !gpu->limits.max_ssbo_size)
        return;
    printf("source_time_tests: LP source-clock API, not VO integration\n");
    fflush(stdout);
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    pl_source_peak_state state = pl_source_peak_state_create(gpu->log, gpu);
    pl_dispatch dispatch = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(rr && state && dispatch);
    struct pl_peak_detect_params peak = pl_peak_detect_default_params;
    peak.smoothing_period = 3;
    peak.scene_threshold_low = peak.scene_threshold_high = 0;
    peak.percentile = 97;
    peak.black_cutoff = 2;
    peak.allow_delayed = false;
    float pixels[32 * 32 * 4];
    const float levels[] = {.1f, 4, .2f};
    const uint64_t all_ids[] = {10, 40, 90}; // gaps are not missing samples
    pl_tex textures[3] = {0};
    struct pl_frame native[3];
    struct custom_mix_owner owners[3] = {0};
    struct sh_peak_result raw[3], after[3], history = {0};
    for (int n = 0; n < 3; n++) {
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int i = 4 * (32*y+x);
                for (int c = 0; c < 3; c++)
                    pixels[i+c] = x < 16 && y < 16 ? levels[n] : 0;
                pixels[i+3] = 1;
            }
        }
        textures[n] = pl_tex_create(gpu, pl_tex_params(
            .w = 32, .h = 32, .format = fmt, .sampleable = true,
            .initial_data = pixels));
        REQUIRE(textures[n]);
        native[n] = custom_mix_frame(textures[n]);
        native[n].color = pl_color_space_hdr10;
        native[n].color.transfer = PL_COLOR_TRC_LINEAR_HDR;
        native[n].acquire = custom_mix_acquire;
        native[n].release = custom_mix_release;
        native[n].user_data = &owners[n];
        pl_shader_obj measurement = NULL;
        pl_shader sh = peak_record_shader(dispatch, textures[n], 1);
        REQUIRE(sh_measure_peak(sh, native[n].color, &measurement, &peak));
        REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
            .shader = &sh, .width = 32, .height = 32)));
        REQUIRE(sh_get_peak_measurement(measurement, true, &raw[n]) == SH_PEAK_READY);
        sh_peak_smooth(&history, &raw[n], &peak);
        after[n] = history;
        pl_shader_obj_destroy(&measurement);
    }
    pl_tex dst = pl_tex_create(gpu, pl_tex_params(
        .w = 32, .h = 32, .format = fmt, .renderable = true,
        .host_readable = true, .blit_dst = true));
    REQUIRE(dst);
    struct pl_frame target = custom_mix_frame(dst);
    struct pl_frame views[3] = {native[0], native[1], native[2]};
    const struct pl_frame *frames[3] = {&views[0], &views[2], &views[2]};
    uint64_t ids[3] = {10, 90, 90}, signatures[3] = {700, 1100, 1100};
    double timestamps[3] = {-.25, 1.75, 2.75};
    struct pl_frame_mix mix = {
        .num_frames = 2, .frames = frames, .signatures = signatures,
        .timestamps = timestamps, .vsync_duration = 1,
    };
    struct custom_mix_test custom = {
        .gpu = gpu, .dispatch = dispatch, .count = 2, .timestamps = timestamps,
    };
    struct pl_custom_frame_mixer mixer = {.priv = &custom, .mix = custom_mix_callback};
    struct pl_source_peak_context context = {
        .state = state, .epoch = 1, .ids = ids, .current_id = 10,
    };
    struct source_time_info info = {0};
    struct pl_color_map_params mapping = pl_color_map_default_params;
    mapping.mastering_clip = 0;
    struct pl_render_params params = pl_render_default_params;
    params.peak_detect_params = &peak;
    params.color_map_params = &mapping;
    params.dither_params = NULL;
    params.skip_caching_single_frame = true;
    params.frame_mixer = NULL;
    params.info_callback = source_time_info_cb;
    params.info_priv = &info;
    // Submit A and C first, then discover B without ever presenting past A.
    for (int n = 0; n < 3; n += 2) {
        REQUIRE(pl_source_peak_submit(rr, state, 1, all_ids[n], &native[n], &peak) >= 0);
        REQUIRE(owners[n].acquired == 1 && owners[n].released == 1);
        REQUIRE(pl_source_peak_submit(rr, state, 1, all_ids[n], &native[n], &peak) >= 0);
        REQUIRE(owners[n].acquired == 1 && owners[n].released == 1);
    }
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
    REQUIRE(info.frames > 0 && info.blends == 0 && custom.calls == 0);
    source_time_expect(state, 1, 10, &after[0]);
    // Nearest-neighbour A must not wait for or latch the unused future C.
    struct pl_hdr_metadata unused = {0};
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, 90, &unused));
    int future_acquired = owners[2].acquired;
    REQUIRE(pl_source_peak_submit(rr, state, 1, 90, &native[2], &peak) ==
            PL_SOURCE_PEAK_PENDING);
    REQUIRE(owners[2].acquired == future_acquired &&
            owners[2].released == future_acquired);
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    REQUIRE(pl_source_peak_submit(rr, state, 1, 90, &native[2], &peak) ==
            PL_SOURCE_PEAK_READY);
    struct sh_peak_result ac = after[0];
    sh_peak_smooth(&ac, &raw[2], &peak);
    source_time_expect(state, 1, 90, &ac);
    uint64_t c_view = custom.ids[1];
    REQUIRE(pl_source_peak_submit(rr, state, 1, 40, &native[1], &peak) >= 0);
    REQUIRE(owners[1].acquired == 1 && owners[1].released == 1);
    frames[1] = &views[1]; frames[2] = &views[2];
    ids[1] = 40; ids[2] = 90;
    signatures[1] = 900; // C retains its render signature across insertion
    mix.num_frames = custom.count = 3;
    timestamps[1] = .75;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    source_time_expect(state, 1, 10, &after[0]);
    source_time_expect(state, 1, 90, &after[2]);
    REQUIRE(custom.ids[2] != c_view);
    custom.blend = true;
    for (int phase = 0; phase < 2; phase++) {
        custom.alpha = phase ? .75f : .25f;
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
        source_time_expect(state, 1, 10, &after[0]);
    }
    custom.blend = false;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
    // View crop is applied after native measurement. Rebuilds and parameter
    // changes must not submit another temporal sample or consume native output.
    struct pl_color_adjustment adjustment = pl_color_adjustment_neutral;
    adjustment.brightness = .02f;
    for (int change = 0; change < 6; change++) {
        switch (change) {
        case 0: pl_renderer_flush_cache(rr); break;
        case 1: pl_renderer_reset(rr); break;
        case 2: target.color.transfer = PL_COLOR_TRC_BT_1886; break;
        case 3: target.color.hdr.max_luma = 100; break;
        case 4: views[0].crop = (pl_rect2df){2, 1, 30, 31}; break;
        case 5: params.color_adjustment = &adjustment; break;
        }
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
        source_time_expect(state, 1, 10, &after[0]);
        for (int n = 0; n < 3; n++) {
            int count = owners[n].acquired;
            REQUIRE(pl_source_peak_submit(rr, state, 1, all_ids[n], &native[n], &peak) >= 0);
            REQUIRE(owners[n].acquired == count && owners[n].released == count);
        }
    }
    mix.num_frames = custom.count = 1;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    mix.num_frames = custom.count = 3;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    source_time_expect(state, 1, 90, &after[2]);
    // A failed attempt to present B must not commit/prune A.
    context.current_id = 40;
    custom.fail = true;
    REQUIRE(!pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    custom.fail = false;
    context.current_id = 10;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    source_time_expect(state, 1, 10, &after[0]);
    uint64_t old_view = custom.ids[0];
    peak.allow_delayed = true;
    REQUIRE(pl_source_peak_submit(rr, state, 1, 10, &native[0], &peak) >= 0);
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    REQUIRE(custom.ids[0] != old_view);
    struct pl_hdr_metadata empty = {0};
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, 10, &empty));
    source_time_expect(state, 1, 40, &after[0]);
    source_time_expect(state, 1, 90, &after[1]);
    peak.allow_delayed = false;
    REQUIRE(pl_source_peak_submit(rr, state, 1, 10, &native[0], &peak) >= 0);
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    source_time_expect(state, 1, 10, &after[0]);
    context.current_id = 40;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    source_time_expect(state, 1, 40, &after[1]);
    // After advancing, render a suffix: retired IDs need not remain queryable.
    frames[0] = &views[2]; ids[0] = 90; signatures[0] = 1200;
    timestamps[0] = -.25; mix.num_frames = custom.count = 1;
    context.current_id = 90;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
    source_time_expect(state, 1, 90, &after[2]);
    REQUIRE(pl_source_peak_state_reset(state, 2));
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, 90, &empty));
    context.epoch = 2; context.current_id = ids[0] = 10; frames[0] = &views[0];
    int acquired = owners[0].acquired;
    REQUIRE(pl_source_peak_submit(rr, state, 2, 10, &native[0], &peak) >= 0);
    REQUIRE(owners[0].acquired == acquired + 1 && owners[0].released == acquired + 1);
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
    source_time_expect(state, 2, 10, &after[0]);
    REQUIRE(custom.ids[0] != old_view);

    struct custom_mix_owner failure = {.fail = true};
    struct pl_frame bad = native[0];
    bad.user_data = &failure;
    REQUIRE(pl_source_peak_submit(rr, state, 2, 100, &bad, &peak) == PL_SOURCE_PEAK_ERROR);
    REQUIRE(failure.acquired == 1 && failure.released == 1);
    failure.fail = false;
    REQUIRE(pl_source_peak_submit(rr, state, 2, 100, &bad, &peak) == PL_SOURCE_PEAK_ERROR);
    REQUIRE(failure.acquired == 1 && failure.released == 1);
    REQUIRE(pl_source_peak_state_reset(state, 3));
    bad.color.transfer = PL_COLOR_TRC_HLG;
    REQUIRE(pl_source_peak_submit(rr, state, 3, 10, &bad, &peak) == PL_SOURCE_PEAK_UNSUPPORTED);
    bad = native[0];
    bad.field = bad.first_field = PL_FIELD_TOP;
    REQUIRE(pl_source_peak_submit(rr, state, 3, 20, &bad, &peak) == PL_SOURCE_PEAK_PENDING);
    acquired = owners[0].acquired;
    REQUIRE(pl_source_peak_submit(rr, state, 4, 10, NULL, NULL) == PL_SOURCE_PEAK_READY);
    REQUIRE(pl_source_peak_submit(rr, state, 4, 10, &native[0], NULL) == PL_SOURCE_PEAK_READY);
    REQUIRE(pl_source_peak_submit(rr, state, 4, 20, NULL, NULL) == PL_SOURCE_PEAK_READY);
    REQUIRE(pl_source_peak_submit(rr, state, 4, 20, NULL, NULL) == PL_SOURCE_PEAK_READY);
    REQUIRE(owners[0].acquired == acquired && owners[0].released == acquired);
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 3, 10, &empty));
    // Invalid enabled submissions cannot change policy, epoch or retained IDs.
    REQUIRE(pl_source_peak_submit(rr, state, 4, 10, NULL, &peak) == PL_SOURCE_PEAK_ERROR);
    REQUIRE(pl_source_peak_submit(rr, state, 99, 30, NULL, &peak) == PL_SOURCE_PEAK_ERROR);
    REQUIRE(pl_source_peak_submit(rr, state, 4, 0, NULL, NULL) == PL_SOURCE_PEAK_ERROR);
    struct pl_render_params disabled_params = params;
    disabled_params.peak_detect_params = NULL;
    uint64_t disabled_id = 10;
    struct pl_source_peak_context disabled_context = {
        .state = state, .epoch = 4, .ids = &disabled_id, .current_id = 10,
    };
    REQUIRE(pl_render_image_source(rr, &native[0], &target, &disabled_params,
                                   &disabled_context));
    REQUIRE(owners[0].acquired == acquired + 1 && owners[0].released == acquired + 1);
    REQUIRE(!pl_renderer_get_hdr_metadata(rr, &empty));
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 4, 10, &empty));
    disabled_id = 20;
    int disabled_acquired = owners[1].acquired;
    REQUIRE(pl_render_image_source(rr, &native[1], &target, &disabled_params,
                                   &disabled_context));
    REQUIRE(owners[1].acquired == disabled_acquired + 1 &&
            owners[1].released == disabled_acquired + 1);
    REQUIRE(!pl_renderer_get_hdr_metadata(rr, &empty));
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 4, 20, &empty));
    // Enabling measurement for the same ID must acquire the actual source.
    acquired = owners[0].acquired;
    REQUIRE(pl_source_peak_submit(rr, state, 4, 10, &native[0], &peak) >= 0);
    REQUIRE(owners[0].acquired == acquired + 1 && owners[0].released == acquired + 1);
    REQUIRE(pl_source_peak_submit(rr, state, 4, 10, &native[0], &peak) >= 0);
    REQUIRE(owners[0].acquired == acquired + 1 && owners[0].released == acquired + 1);
    disabled_id = 10;
    REQUIRE(pl_render_image_source(rr, &native[0], &target, &params, &disabled_context));
    REQUIRE(owners[0].acquired == acquired + 2 && owners[0].released == acquired + 2);
    source_time_expect(state, 4, 10, &after[0]);
    REQUIRE(pl_source_peak_state_reset(state, 5));
    struct source_time_reentry reentry = {
        .rr = rr, .state = state, .source = &native[0], .params = &peak,
    };
    bad = native[0];
    bad.acquire = source_time_acquire;
    bad.release = source_time_release;
    bad.user_data = &reentry;
    REQUIRE(pl_source_peak_submit(rr, state, 5, 10, &bad, &peak) >= 0);
    REQUIRE(reentry.acquired == 1 && reentry.released == 1);
    REQUIRE(pl_source_peak_submit(rr, state, 5, 10, &bad, &peak) >= 0);
    REQUIRE(reentry.acquired == 1 && reentry.released == 1);
    context.epoch = 5;
    REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
    source_time_expect(state, 5, 10, &after[0]);
    bad = native[0];
    bad.profile = (struct pl_icc_profile){.data = "invalid ICC", .len = 11};
    acquired = owners[0].acquired;
    REQUIRE(pl_source_peak_submit(rr, state, 5, 100, &bad, &peak) == PL_SOURCE_PEAK_UNSUPPORTED);
    REQUIRE(owners[0].acquired == acquired + 1 && owners[0].released == acquired + 1);
    source_time_expect(state, 5, 10, &after[0]);
    // Oversample uses B and C. A failed used-source measurement must not be
    // swallowed by the legacy frame-mixer fallback or commit current B.
    REQUIRE(pl_source_peak_state_reset(state, 6));
    REQUIRE(pl_source_peak_submit(rr, state, 6, 10, &native[0], &peak) >= 0);
    REQUIRE(pl_source_peak_submit(rr, state, 6, 40, &native[1], &peak) >= 0);
    struct custom_mix_owner failed_future = {.fail = true};
    struct pl_frame failed_source = native[2];
    failed_source.user_data = &failed_future;
    REQUIRE(pl_source_peak_submit(rr, state, 6, 90, &failed_source, &peak) ==
            PL_SOURCE_PEAK_ERROR);
    REQUIRE(failed_future.acquired == 1 && failed_future.released == 1);
    const struct pl_frame *failure_frames[] = {&native[0], &native[1], &failed_source};
    uint64_t failure_ids[] = {10, 40, 90};
    uint64_t failure_signatures[] = {1800, 1801, 1802};
    double failure_times[] = {-1.25, -.25, .75};
    struct pl_frame_mix failure_mix = {
        .num_frames = 3, .frames = failure_frames, .signatures = failure_signatures,
        .timestamps = failure_times, .vsync_duration = 1,
    };
    struct pl_source_peak_context failure_context = {
        .state = state, .epoch = 6, .ids = failure_ids, .current_id = 40,
    };
    struct pl_render_params failure_params = params;
    failure_params.frame_mixer = &pl_filter_oversample;
    REQUIRE(!(pl_renderer_get_errors(rr).errors & PL_RENDER_ERR_FRAME_MIXING));
    REQUIRE(!pl_render_image_mix_source(rr, &failure_mix, &target, &failure_params,
                                       &failure_context, NULL));
    REQUIRE(!(pl_renderer_get_errors(rr).errors & PL_RENDER_ERR_FRAME_MIXING));
    REQUIRE(failed_future.acquired == 1 && failed_future.released == 1);
    REQUIRE(pl_source_peak_submit(rr, state, 6, 90, &failed_source, &peak) ==
            PL_SOURCE_PEAK_ERROR);
    REQUIRE(failed_future.acquired == 1 && failed_future.released == 1);
    failure_mix.num_frames = 1;
    failure_times[0] = 0;
    failure_context.current_id = 10;
    failure_params.frame_mixer = NULL;
    REQUIRE(pl_render_image_mix_source(rr, &failure_mix, &target, &failure_params,
                                      &failure_context, NULL));
    source_time_expect(state, 6, 10, &after[0]);
    // Logical source events are independent of visible phase/nearest source.
    // Retention is explicit: shrinking a mix must not retire older snapshots.
    REQUIRE(pl_source_peak_state_reset(state, 7));
    for (int n = 0; n < 3; n++)
        REQUIRE(pl_source_peak_submit(rr, state, 7, all_ids[n], &native[n], &peak) >= 0);
    const struct pl_frame *clock_frames[] = {&native[2], &native[2]};
    uint64_t clock_ids[] = {90, 90};
    uint64_t clock_signatures[] = {1902, 1902};
    double clock_times[] = {0, .75};
    struct pl_frame_mix clock_mix = {
        .num_frames = 1, .frames = clock_frames, .signatures = clock_signatures,
        .timestamps = clock_times, .vsync_duration = 1,
    };
    struct pl_source_peak_context clock_context = {
        .state = state, .epoch = 7, .ids = clock_ids, .current_id = 10,
        .retire_before_id = 0,
    };
    REQUIRE(pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                      &clock_context, NULL));
    source_time_expect(state, 7, 10, &after[0]);
    source_time_expect(state, 7, 90, &after[2]);
    clock_frames[0] = &native[0];
    clock_ids[0] = 10;
    clock_signatures[0] = 1900;
    REQUIRE(pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                      &clock_context, NULL));
    source_time_expect(state, 7, 10, &after[0]);
    // Advancing the logical event to B may still display A (negative offset),
    // without forcing a rollback of the source clock.
    clock_context.current_id = 40;
    clock_frames[0] = &native[2];
    clock_ids[0] = 90;
    clock_signatures[0] = 1902;
    REQUIRE(pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                      &clock_context, NULL));
    // A is now committed, but a future-only view is not a retirement request.
    source_time_expect(state, 7, 10, &after[0]);
    clock_frames[0] = &native[0];
    clock_ids[0] = 10;
    clock_signatures[0] = 1900;
    REQUIRE(pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                      &clock_context, NULL));
    source_time_expect(state, 7, 40, &after[1]);
    source_time_expect(state, 7, 10, &after[0]);
    clock_context.retire_before_id = 40; // legal relative to current, not visible A
    REQUIRE(!pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                       &clock_context, NULL));
    source_time_expect(state, 7, 10, &after[0]);
    clock_mix.num_frames = 1;
    clock_frames[0] = &native[2];
    clock_ids[0] = 90;
    clock_signatures[0] = 1902;
    clock_times[0] = 0;
    clock_context.retire_before_id = 50; // <= visible C, but greater than current B
    REQUIRE(!pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                       &clock_context, NULL));
    clock_frames[0] = &native[1];
    clock_ids[0] = 40;
    clock_signatures[0] = 1901;
    clock_times[0] = -.25;
    clock_mix.num_frames = 2;
    // Failure while attempting current C must neither advance from B nor
    // retire A, even though the requested retirement boundary is valid.
    struct custom_mix_test clock_custom = {
        .gpu = gpu, .dispatch = dispatch, .count = 2,
        .timestamps = clock_times, .fail = true,
    };
    struct pl_custom_frame_mixer clock_mixer = {
        .priv = &clock_custom, .mix = custom_mix_callback,
    };
    clock_context.current_id = 90;
    clock_context.retire_before_id = 40;
    REQUIRE(!pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                       &clock_context, &clock_mixer));
    REQUIRE(clock_custom.calls == 1);
    source_time_expect(state, 7, 10, &after[0]);
    clock_context.current_id = 40;
    clock_context.retire_before_id = 0;
    clock_mix.num_frames = 1;
    clock_frames[0] = &native[0];
    clock_ids[0] = 10;
    clock_signatures[0] = 1900;
    clock_times[0] = 0;
    REQUIRE(pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                      &clock_context, NULL));
    source_time_expect(state, 7, 10, &after[0]);
    clock_frames[0] = &native[1];
    clock_ids[0] = 40;
    clock_signatures[0] = 1901;
    clock_times[0] = -.25;
    clock_mix.num_frames = 2;
    clock_custom.fail = false;
    clock_context.retire_before_id = 40;
    REQUIRE(pl_render_image_mix_source(rr, &clock_mix, &target, &params,
                                      &clock_context, &clock_mixer));
    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 7, 10, &empty));
    source_time_expect(state, 7, 40, &after[1]);
    source_time_expect(state, 7, 90, &after[2]);
    source_time_pixels(gpu, rr, dispatch, dst, &native[0], &native[1]);
    source_time_image(gpu, dst, native, after);
    source_time_hlg_views(gpu, dispatch, dst);
    pl_renderer_destroy(&rr);
    pl_source_peak_state_destroy(&state);
    pl_tex_destroy(gpu, &custom.output);
    pl_dispatch_destroy(&dispatch);
    for (int n = 0; n < 3; n++) pl_tex_destroy(gpu, &textures[n]);
    pl_tex_destroy(gpu, &dst);
    printf("- source-time ownership, prefixes, delay, failure and epochs passed\n");
}

static void source_time_boundary_tests(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fmt || !gpu->glsl.compute || !gpu->limits.max_ssbo_size) {
        puts("Source boundary GPU capability unavailable: not run");
        return;
    }
    const uint64_t ids[] = {10, 20, 30, 40, 45, 50, 60};
    const bool hlg[] = {false, true, true, false, false, true, true};
    const float level[] = {.1f, .85f, 1.0f, .2f, .4f, .95f, .8f};
    float pixels[32 * 32 * 4], actual[32 * 32 * 4], expected[32 * 32 * 4];
    pl_tex textures[7] = {0};
    struct pl_frame sources[7];
    struct custom_mix_owner owners[7] = {0};
    for (int n = 0; n < 7; n++) {
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                for (int c = 0; c < 3; c++)
                    pixels[(y * 32 + x) * 4 + c] =
                        level[n] * (.1f + .9f * x / 31);
                pixels[(y * 32 + x) * 4 + 3] = 1;
            }
        }
        textures[n] = pl_tex_create(gpu, pl_tex_params(
            .w = 32, .h = 32, .format = fmt, .sampleable = true,
            .initial_data = pixels));
        REQUIRE(textures[n]);
        sources[n] = custom_mix_frame(textures[n]);
        sources[n].color = pl_color_space_hdr10;
        sources[n].color.transfer = hlg[n] ? PL_COLOR_TRC_HLG : PL_COLOR_TRC_LINEAR_HDR;
        sources[n].color.hdr.max_luma = 1000;
        sources[n].acquire = custom_mix_acquire;
        sources[n].release = custom_mix_release;
        sources[n].user_data = &owners[n];
    }
    pl_tex dst = pl_tex_create(gpu, pl_tex_params(
        .w = 32, .h = 32, .format = fmt, .renderable = true, .host_readable = true));
    REQUIRE(dst);
    struct pl_frame target = custom_mix_frame(dst);
    // Keep both HLG segments above the display range, so smoothing changes
    // the tone curve rather than comparing two uncompressed SDR-range views.
    target.color.hdr.max_luma = 50;
    pl_dispatch dispatch = pl_dispatch_create(gpu->log, gpu);
    REQUIRE(dispatch);
    for (int delayed = 0; delayed < 2; delayed++) {
        pl_renderer rr = pl_renderer_create(gpu->log, gpu);
        pl_source_peak_state state = pl_source_peak_state_create(gpu->log, gpu);
        REQUIRE(rr && state);
        struct pl_peak_detect_params peak = pl_peak_detect_default_params;
        peak.smoothing_period = 3;
        peak.scene_threshold_low = peak.scene_threshold_high = 0;
        peak.percentile = 97;
        peak.black_cutoff = 2;
        peak.allow_delayed = delayed;
        struct pl_color_map_params mapping = pl_color_map_default_params;
        mapping.mastering_clip = 0;
        struct pl_render_params params = pl_render_default_params;
        params.peak_detect_params = &peak;
        params.color_map_params = &mapping;
        params.dither_params = NULL;
        params.frame_mixer = NULL;
        params.skip_caching_single_frame = true;
        for (int n = 0; n < 7; n++) {
            enum pl_source_peak_status status = pl_source_peak_submit(
                rr, state, 1, ids[n], &sources[n], &peak);
            REQUIRE(hlg[n] ? status == PL_SOURCE_PEAK_UNSUPPORTED : status >= 0);
            int acquired = owners[n].acquired;
            REQUIRE(pl_source_peak_submit(rr, state, 1, ids[n], &sources[n], &peak) == status);
            REQUIRE(owners[n].acquired == acquired && owners[n].released == acquired);
        }
        // Independent raw C measurement defines the history after the boundary.
        pl_shader_obj measurement = NULL;
        pl_shader sh = peak_record_shader(dispatch, textures[3], 1);
        REQUIRE(sh_measure_peak(sh, sources[3].color, &measurement, &peak));
        REQUIRE(pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
            .shader = &sh, .width = 32, .height = 32)));
        struct sh_peak_result raw, cold = {0};
        REQUIRE(sh_get_peak_measurement(measurement, true, &raw) == SH_PEAK_READY);
        sh_peak_smooth(&cold, &raw, &peak);
        pl_shader_obj_destroy(&measurement);
        const struct pl_frame *visible = &sources[0];
        uint64_t visible_id = ids[0], signature = 2000;
        double timestamp = 0;
        struct pl_frame_mix mix = {
            .num_frames = 1, .frames = &visible, .signatures = &signature,
            .timestamps = &timestamp, .vsync_duration = 1,
        };
        struct pl_source_peak_context context = {
            .state = state, .epoch = 1, .ids = &visible_id, .current_id = ids[0],
        };
        struct custom_mix_test custom = {
            .gpu = gpu, .dispatch = dispatch, .count = 1, .timestamps = &timestamp,
        };
        struct pl_custom_frame_mixer mixer = {.priv = &custom, .mix = custom_mix_callback};
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
        struct pl_hdr_metadata metadata = {0};
        REQUIRE(pl_source_peak_get_hdr_metadata(state, 1, ids[0], &metadata) == !delayed);
        // Current is unsupported B, but the only used source is eligible C.
        context.current_id = ids[1];
        visible = &sources[3]; visible_id = ids[3]; signature = 2003;
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
        REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, ids[1], &metadata));
        REQUIRE(!pl_renderer_get_hdr_metadata(rr, &metadata));
        if (delayed)
            REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, ids[3], &metadata));
        else
            source_time_expect(state, 1, ids[3], &cold);
        REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
        // Merely being the clock boundary does not prohibit custom eligible C.
        visible = &sources[1]; visible_id = ids[1]; signature = 2001;
        int calls = custom.calls;
        REQUIRE(!pl_render_image_mix_source(rr, &mix, &target, &params, &context, &mixer));
        REQUIRE(custom.calls == calls);
        REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, ids[1], &metadata));
        for (int segment = 0; segment < 2; segment++) {
            // Each oracle starts at this segment, but keeps its own history for
            // the second source. This distinguishes reset-on-entry from reset
            // every frame, without changing any production peak parameters.
            pl_renderer oracle = pl_renderer_create(gpu->log, gpu);
            REQUIRE(oracle);
            for (int step = 0; step < 2; step++) {
                int n = segment ? 5 + step : 1 + step;
                visible = &sources[n]; visible_id = ids[n]; signature = 2000 + n;
                context.current_id = ids[n];
                REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
                custom_mix_download(gpu, dst, actual);
                REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, ids[n], &metadata));
                REQUIRE(pl_render_image(oracle, &sources[n], &target, &params));
                custom_mix_download(gpu, dst, expected);
                printf("Source HLG boundary delayed=%d segment=%d step=%d\n",
                       delayed, segment, step);
                fflush(stdout);
                custom_mix_compare(actual, expected, 32 * 32 * 4);
                // Public timed metadata follows the logical current source,
                // never the legacy detector used to display this HLG segment.
                struct pl_hdr_metadata observed = {0}, wanted = {0};
                REQUIRE(!pl_renderer_get_hdr_metadata(rr, &observed));
                REQUIRE(pl_renderer_get_hdr_metadata(oracle, &wanted));
                REQUIRE(wanted.avg_pq_y > 0 && wanted.max_pq_y > 0);
                if (step) {
                    pl_renderer fresh = pl_renderer_create(gpu->log, gpu);
                    REQUIRE(fresh);
                    REQUIRE(pl_render_image(fresh, &sources[n], &target, &params));
                    float fresh_pixels[32 * 32 * 4];
                    custom_mix_download(gpu, dst, fresh_pixels);
                    struct pl_hdr_metadata fresh_metadata = {0};
                    REQUIRE(pl_renderer_get_hdr_metadata(fresh, &fresh_metadata));
                    REQUIRE(fresh_metadata.avg_pq_y > 0 && fresh_metadata.max_pq_y > 0);
                    REQUIRE(fabsf(wanted.avg_pq_y - fresh_metadata.avg_pq_y) > 1e-6f ||
                            fabsf(wanted.max_pq_y - fresh_metadata.max_pq_y) > 1e-6f);
                    double difference = 0;
                    for (int i = 0; i < 32 * 32 * 4; i++)
                        difference = PL_MAX(difference, fabs(expected[i] - fresh_pixels[i]));
                    printf("Source HLG history observable delayed=%d segment=%d max=%g\n",
                           delayed, segment, difference);
                    fflush(stdout);
                    REQUIRE(difference > .003);
                    pl_renderer_destroy(&fresh);
                }
            }
            pl_renderer_destroy(&oracle);
            if (!segment) {
                visible = &sources[3]; visible_id = ids[3]; signature = 2003;
                context.current_id = ids[3];
                REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
                if (delayed)
                    REQUIRE(!pl_source_peak_get_hdr_metadata(state, 1, ids[3], &metadata));
                else
                    source_time_expect(state, 1, ids[3], &cold);
                visible = &sources[4]; visible_id = ids[4]; signature = 2004;
                context.current_id = ids[4];
                REQUIRE(pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
                if (delayed)
                    source_time_expect(state, 1, ids[4], &cold);
            }
        }
        // A real acquisition ERROR must never become an unsupported boundary.
        struct custom_mix_owner failure = {.fail = true};
        struct pl_frame failed = sources[0];
        failed.user_data = &failure;
        REQUIRE(pl_source_peak_submit(rr, state, 1, 90, &failed, &peak) == PL_SOURCE_PEAK_ERROR);
        visible = &failed; visible_id = context.current_id = 90; signature = 2090;
        REQUIRE(!pl_render_image_mix_source(rr, &mix, &target, &params, &context, NULL));
        REQUIRE(failure.acquired == 1 && failure.released == 1);
        REQUIRE(pl_source_peak_submit(rr, state, 1, 90, &failed, &peak) == PL_SOURCE_PEAK_ERROR);
        REQUIRE(failure.acquired == 1 && failure.released == 1);
        pl_renderer_destroy(&rr);
        pl_source_peak_state_destroy(&state);
    }
    pl_dispatch_destroy(&dispatch);
    for (int n = 0; n < 7; n++)
        pl_tex_destroy(gpu, &textures[n]);
    pl_tex_destroy(gpu, &dst);
    puts("Source-time unsupported boundaries and legacy segment ownership: passed");
}

static void source_time_field_tests(pl_gpu gpu)
{
    enum { SIZE = 16, SOURCES = 4 };
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 4, 8, 8,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    REQUIRE(fmt);
    uint8_t woven[SIZE * SIZE * 4];
    for (int i = 0; i < SIZE * SIZE; i++) {
        memset(&woven[4 * i], (i / SIZE) % 2 ? 224 : 32, 3);
        woven[4 * i + 3] = 255;
    }
    pl_tex input = pl_tex_create(gpu, pl_tex_params(
        .w = SIZE, .h = SIZE, .format = fmt, .sampleable = true,
        .initial_data = woven));
    pl_tex dst = pl_tex_create(gpu, pl_tex_params(
        .w = SIZE, .h = SIZE, .format = fmt, .renderable = true,
        .host_readable = true));
    REQUIRE(input && dst);
    struct pl_frame target = custom_mix_frame(dst);
    const struct pl_render_params params = {
        .frame_mixer = &pl_filter_mitchell_clamp,
        .deinterlace_params = pl_deinterlace_params(.algo = PL_DEINTERLACE_BOB),
    };
    pl_renderer timed = pl_renderer_create(gpu->log, gpu);
    pl_renderer ordinary = pl_renderer_create(gpu->log, gpu);
    pl_queue queue = pl_queue_create(gpu);
    pl_source_peak_state state = pl_source_peak_state_create(gpu->log, gpu);

    for (enum pl_field first = PL_FIELD_TOP; first <= PL_FIELD_BOTTOM; first++) {
        pl_queue_reset(queue);
        REQUIRE(pl_source_peak_state_reset(state, first));
        struct pl_frame native[SOURCES];
        uint64_t source_ids[SOURCES];
        for (int i = 0; i < SOURCES; i++) {
            source_ids[i] = i + 1;
            native[i] = custom_mix_frame(input);
            native[i].user_data = &source_ids[i];
            REQUIRE(pl_source_peak_submit(timed, state, first, source_ids[i], NULL, NULL) ==
                    PL_SOURCE_PEAK_READY);
            pl_queue_push(queue, &(struct pl_source_frame) {
                .pts = i * .04, .duration = .04, .first_field = first,
                .frame_data = &native[i], .map = frame_passthrough,
            });
        }
        pl_queue_push(queue, NULL);
        struct pl_frame_mix mix;
        REQUIRE(pl_queue_update(queue, &mix, &(struct pl_queue_params) {
            .pts = .01, .radius = pl_frame_mix_radius(&params),
            .vsync_duration = 1.0 / 120.0,
        }) == PL_QUEUE_OK);
        REQUIRE(mix.num_frames >= 2 && mix.num_frames <= 2 * SOURCES);
        uint64_t ids[2 * SOURCES];
        const struct pl_frame *views[2 * SOURCES];
        int pair = -1;
        for (int i = 0; i < mix.num_frames; i++) {
            views[i] = mix.frames[i];
            ids[i] = *(const uint64_t *) views[i]->user_data;
            if (i && ids[i] == ids[i - 1]) {
                REQUIRE(views[i - 1]->field == first && views[i]->field == pl_field_other(first));
                REQUIRE(mix.signatures[i] != mix.signatures[i - 1]);
                REQUIRE(mix.timestamps[i] > mix.timestamps[i - 1]);
                pair = i;
            }
        }
        REQUIRE(pair > 0);
        struct pl_source_peak_context context = {
            .state = state, .epoch = first, .ids = ids, .current_id = ids[0],
        };
        uint8_t expected[sizeof(woven)], actual[sizeof(woven)];
        REQUIRE(pl_render_image_mix(ordinary, &mix, &target, &params));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = expected)));
        REQUIRE(pl_render_image_mix_source(timed, &mix, &target, &params, &context, NULL));
        REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = dst, .ptr = actual)));
        REQUIRE(memcmp(expected, actual, sizeof(actual)) == 0);

        views[pair] = views[pair - 1];
        mix.frames = views;
        REQUIRE(!pl_render_image_mix_source(timed, &mix, &target, &params, &context, NULL));
        pl_queue_reset(queue);
    }
    pl_queue_destroy(&queue);
    pl_renderer_destroy(&ordinary);
    pl_renderer_destroy(&timed);
    pl_source_peak_state_destroy(&state);
    pl_tex_destroy(gpu, &dst);
    pl_tex_destroy(gpu, &input);
    puts("Source-time field pairs retain the ordinary renderer output");
}

static void snapshot_rejected(pl_dispatch dp, pl_shader src)
{
    pl_shader dst = pl_dispatch_begin(dp);
    REQUIRE(sh_require(dst, PL_SHADER_SIG_NONE, 0, 0));
    REQUIRE(!sh_can_snapshot(src));
    REQUIRE(!sh_snapshot(src, dst));
    REQUIRE(dst->mutable && !dst->failed && dst->output == PL_SHADER_SIG_COLOR);
    pl_dispatch_abort(dp, &dst);
}

static int snapshot_destroyed;
static void snapshot_uninit(pl_gpu gpu, void *priv)
{
    snapshot_destroyed++;
}

static void snapshot_read(pl_gpu gpu, pl_tex tex, float expected)
{
    float pixels[16 * 16 * 4];
    REQUIRE(pl_tex_download(gpu, pl_tex_transfer_params(.tex = tex, .ptr = pixels)));
    for (int i = 0; i < 16 * 16; i++)
        REQUIRE_FEQ(pixels[4*i], expected, 1e-6);
}

struct snapshot_hook {
    pl_buf counter;
    int hooks, acquires, releases;
};

static bool snapshot_acquire(pl_gpu gpu, struct pl_frame *frame)
{
    struct snapshot_hook *test = frame->user_data;
    test->acquires++;
    return true;
}

static void snapshot_release(pl_gpu gpu, struct pl_frame *frame)
{
    struct snapshot_hook *test = frame->user_data;
    test->releases++;
}

static struct pl_hook_res snapshot_hook(void *priv,
                                        const struct pl_hook_params *params)
{
    struct snapshot_hook *test = priv;
    test->hooks++;
    REQUIRE(params->rect.x0 == 0 && params->rect.y0 == 0 &&
            params->rect.x1 == 16 && params->rect.y1 == 16);
    struct pl_buffer_var var = {.var = pl_var_uint("counter")};
    var.layout = pl_std430_layout(0, &var.var);
    struct pl_shader_desc desc = {
        .desc = {.name = "counter_buffer", .type = PL_DESC_BUF_STORAGE,
                 .access = PL_DESC_ACCESS_READWRITE},
        .binding.object = test->counter,
        .buffer_vars = &var, .num_buffer_vars = 1,
    };
    REQUIRE(pl_shader_custom(params->sh, &(struct pl_custom_shader) {
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .output_w = 16, .output_h = 16,
        .body = "atomicAdd(counter, 1u);\n",
        .descriptors = &desc, .num_descriptors = 1,
    }));
    REQUIRE(!sh_can_snapshot(params->sh));
    return (struct pl_hook_res) {
        .output = PL_HOOK_SIG_COLOR, .sh = params->sh,
        .repr = params->repr, .color = params->color,
        .components = params->components, .rect = params->rect,
    };
}

static void shader_snapshot_tests(pl_gpu gpu)
{
    puts("- shader snapshot ownership");
    fflush(stdout);
    snapshot_destroyed = 0;
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 32,
        PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!fmt || !gpu->glsl.compute || gpu->limits.max_ssbo_size < 4) {
        puts("shader_snapshot_tests: skipped (FP32 and compute storage required)");
        return;
    }
    float pixels[16 * 16 * 4];
    for (int i = 0; i < 16 * 16; i++) {
        pixels[4*i] = pixels[4*i+1] = pixels[4*i+2] = 0.25f;
        pixels[4*i+3] = 1;
    }
    pl_tex source = pl_tex_create(gpu, pl_tex_params(
        .w = 16, .h = 16, .format = fmt, .sampleable = true,
        .host_writable = true, .initial_data = pixels));
    pl_tex target = pl_tex_create(gpu, pl_tex_params(
        .w = 16, .h = 16, .format = fmt, .renderable = true, .host_readable = true));
    REQUIRE(source && target);
    pl_dispatch dp = pl_dispatch_create(gpu->log, gpu);
    pl_shader sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(.tex = source)));
    ident_t gain = sh_var_float(sh, "gain", 0.25f, true);
    GLSL("color.r += "$";\n", gain);
    pl_shader_obj obj = NULL;
    REQUIRE(sh_require_obj(sh, &obj, PL_SHADER_OBJ_LUT, sizeof(int),
                           snapshot_uninit));
    REQUIRE(sh_can_snapshot(sh));
    pl_shader copy = pl_dispatch_begin(dp);
    REQUIRE(!sh_snapshot(sh, sh) && !sh_snapshot(sh, NULL));
    REQUIRE(sh_snapshot(sh, copy));
    REQUIRE(copy && sh->mutable && copy->mutable);
    REQUIRE(sh->vars.num == copy->vars.num && sh->vas.num == copy->vas.num);
    for (int i = 0; i < sh->vars.num; i++)
        REQUIRE(sh->vars.elem[i].data != copy->vars.elem[i].data);
    for (int i = 0; i < sh->vas.num; i++)
        for (int j = 0; j < 4; j++)
            REQUIRE(sh->vas.elem[i].data[j] != copy->vas.elem[i].data[j]);
    float replacement = 0.5f;
    memcpy((void *) sh->vars.elem[sh->vars.num-1].data, &replacement,
           sizeof(replacement));
    GLSL("color.r += 0.125;\n");
    pl_shader_obj_destroy(&obj);
    REQUIRE(snapshot_destroyed == 0);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &sh, .target = target)));
    snapshot_read(gpu, target, 0.875f);
    REQUIRE(snapshot_destroyed == 0);
    REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(.shader = &copy, .target = target)));
    snapshot_read(gpu, target, 0.5f);
    pl_dispatch_destroy(&dp);
    REQUIRE(snapshot_destroyed == 1);

    puts("- shader snapshot bounded-dispatch-pool");
    fflush(stdout);
    dp = pl_dispatch_create(gpu->log, gpu);
    pl_shader pooled[2] = {0};
    for (int n = 0; n < 32; n++) {
        sh = pl_dispatch_begin(dp);
        copy = pl_dispatch_begin(dp);
        REQUIRE(sh != copy);
        if (!n) {
            pooled[0] = sh;
            pooled[1] = copy;
        } else {
            REQUIRE((sh == pooled[0] || sh == pooled[1]) &&
                    (copy == pooled[0] || copy == pooled[1]));
        }
        REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(.tex = source)));
        obj = NULL;
        REQUIRE(sh_require_obj(sh, &obj, PL_SHADER_OBJ_LUT, sizeof(int),
                               snapshot_uninit));
        REQUIRE(sh_snapshot(sh, copy));
        pl_shader_obj_destroy(&obj);
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &sh, .target = target)));
        REQUIRE(snapshot_destroyed == n + 1);
        REQUIRE(pl_dispatch_finish(dp, pl_dispatch_params(
            .shader = &copy, .target = target)));
        REQUIRE(snapshot_destroyed == n + 2);
    }
    snapshot_read(gpu, target, 0.25f);
    pl_dispatch_destroy(&dp);
    REQUIRE(snapshot_destroyed == 33);

    puts("- shader snapshot custom-subpass");
    fflush(stdout);
    dp = pl_dispatch_create(gpu->log, gpu);
    sh = pl_dispatch_begin_ex(dp, true);
    pl_shader sub = pl_dispatch_begin_ex(dp, true);
    REQUIRE(pl_shader_sample_nearest(sub, pl_sample_src(.tex = source)));
    REQUIRE(pl_shader_custom(sub, &(struct pl_custom_shader) {
        .input = PL_SHADER_SIG_COLOR, .output = PL_SHADER_SIG_COLOR,
        .body = "color.rgb *= 1.0;\n",
    }));
    snapshot_rejected(dp, sub);
    ident_t subpass = sh_subpass(sh, sub);
    REQUIRE(subpass);
    REQUIRE(sh_require(sh, PL_SHADER_SIG_NONE, 0, 0));
    GLSL("vec4 color = "$"();\n", subpass);
    REQUIRE(sh->mutable && !sh->failed && sh->type == SH_AUTO);
    snapshot_rejected(dp, sh);
    pl_dispatch_abort(dp, &sub);
    pl_dispatch_abort(dp, &sh);
    pl_dispatch_destroy(&dp);

    puts("- shader snapshot random-dither");
    fflush(stdout);
    dp = pl_dispatch_create(gpu->log, gpu);
    for (int temporal = 0; temporal < 2; temporal++) {
        sh = pl_dispatch_begin(dp);
        REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(.tex = source)));
        REQUIRE(sh_prng(sh, temporal, NULL));
        snapshot_rejected(dp, sh);
        pl_dispatch_abort(dp, &sh);
    }
    sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(.tex = source)));
    pl_shader_obj dither = NULL;
    pl_shader_dither(sh, 8, &dither, &pl_dither_default_params);
    REQUIRE(!sh->failed);
    snapshot_rejected(dp, sh);
    pl_dispatch_abort(dp, &sh);
    pl_shader_obj_destroy(&dither);
    pl_dispatch_destroy(&dp);

    puts("- shader snapshot storage-access");
    fflush(stdout);
    uint32_t zero = 0;
    struct snapshot_hook test = {0};
    test.counter = pl_buf_create(gpu, pl_buf_params(
        .size = sizeof(zero), .storable = true, .host_readable = true,
        .initial_data = &zero));
    REQUIRE(test.counter);
    dp = pl_dispatch_create(gpu->log, gpu);
    sh = pl_dispatch_begin(dp);
    REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(.tex = source)));
    struct pl_buffer_var var = {.var = pl_var_uint("value")};
    var.layout = pl_std430_layout(0, &var.var);
    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {.name = "writable", .type = PL_DESC_BUF_STORAGE,
                 .access = PL_DESC_ACCESS_READWRITE},
        .binding.object = test.counter,
        .buffer_vars = &var, .num_buffer_vars = 1,
    });
    snapshot_rejected(dp, sh);
    sh->descs.elem[sh->descs.num-1].desc.access = PL_DESC_ACCESS_READONLY;
    REQUIRE(sh_can_snapshot(sh));
    copy = pl_dispatch_begin(dp);
    REQUIRE(sh_snapshot(sh, copy));
    REQUIRE(copy);
    REQUIRE(copy->descs.elem[copy->descs.num-1].buffer_vars !=
            sh->descs.elem[sh->descs.num-1].buffer_vars);
    pl_dispatch_abort(dp, &copy);
    pl_dispatch_abort(dp, &sh);
    pl_dispatch_destroy(&dp);

    puts("- shader snapshot side-effect-hook");
    fflush(stdout);
    const struct pl_hook hook = {
        .stages = PL_HOOK_RGB, .input = PL_HOOK_SIG_COLOR,
        .priv = &test, .hook = snapshot_hook,
    };
    const struct pl_hook *hooks[] = {&hook};
    struct pl_frame image = custom_mix_frame(source);
    image.color.transfer = PL_COLOR_TRC_PQ;
    image.color.hdr.max_luma = image.color.hdr.max_cll = 1000;
    image.acquire = snapshot_acquire;
    image.release = snapshot_release;
    image.user_data = &test;
    struct pl_frame output = custom_mix_frame(target);
    pl_renderer rr = pl_renderer_create(gpu->log, gpu);
    struct pl_render_params params = pl_render_default_params;
    params.hooks = hooks;
    params.num_hooks = 1;
    params.dither_params = NULL;
    params.peak_detect_params = &pl_peak_detect_default_params;
    REQUIRE(pl_render_image(rr, &image, &output, &params));
    REQUIRE(pl_renderer_get_errors(rr).errors == 0);
    REQUIRE(test.hooks == 1 && test.acquires == 1 && test.releases == 1);
    uint32_t count = 0;
    REQUIRE(pl_buf_read(gpu, test.counter, 0, &count, sizeof(count)));
    REQUIRE(count == 16 * 16);
    printf("snapshot counter=%u acquire=%d release=%d hooks=%d\n",
           count, test.acquires, test.releases, test.hooks);
    pl_renderer_destroy(&rr);

    puts("- shader snapshot same-frame-smoothing");
    fflush(stdout);
    // Compare each frame's smoothed peak with one explicit detector dispatch.
    rr = pl_renderer_create(gpu->log, gpu);
    dp = pl_dispatch_create(gpu->log, gpu);
    pl_shader_obj peak_state = NULL;
    struct pl_peak_detect_params peak = pl_peak_detect_default_params;
    peak.allow_delayed = false;
    peak.smoothing_period = 2;
    peak.scene_threshold_low = peak.scene_threshold_high = 0;
    params.hooks = NULL;
    params.num_hooks = 0;
    params.peak_detect_params = &peak;
    const float levels[] = {0.25f, 0.75f, 0.5f};
    for (int n = 0; n < 3; n++) {
        for (int i = 0; i < 16 * 16; i++)
            for (int c = 0; c < 3; c++)
                pixels[4*i+c] = levels[n];
        REQUIRE(pl_tex_upload(gpu, pl_tex_transfer_params(
            .tex = source, .ptr = pixels)));
        REQUIRE(pl_render_image(rr, &image, &output, &params));
        REQUIRE(pl_renderer_get_errors(rr).errors == 0);
        struct pl_hdr_metadata actual = {0}, expected = {0};
        REQUIRE(pl_renderer_get_hdr_metadata(rr, &actual));
        sh = pl_dispatch_begin(dp);
        REQUIRE(pl_shader_sample_nearest(sh, pl_sample_src(.tex = source)));
        REQUIRE(pl_shader_detect_peak(sh, image.color, &peak_state, &peak));
        REQUIRE(pl_dispatch_compute(dp, pl_dispatch_compute_params(
            .shader = &sh, .width = 16, .height = 16)));
        pl_gpu_finish(gpu);
        REQUIRE(pl_get_detected_hdr_metadata(peak_state, &expected));
        peak_record_metadata_equal(&actual, &expected);
    }
    REQUIRE(test.acquires == 4 && test.releases == 4 && test.hooks == 1);
    pl_shader_obj_destroy(&peak_state);
    pl_dispatch_destroy(&dp);
    pl_renderer_destroy(&rr);
    pl_buf_destroy(gpu, &test.counter);
    pl_tex_destroy(gpu, &source);
    pl_tex_destroy(gpu, &target);
    puts("- shader snapshot ownership and GPU hook passed");
}

void gpu_shader_tests(pl_gpu gpu)
{
    shader_snapshot_tests(gpu);
    pl_buffer_tests(gpu);
    pl_texture_tests(gpu);
    pl_planar_tests(gpu);
    pl_shader_tests(gpu);
#ifdef PL_HAVE_DOVI
    pl_dovi_tests(gpu);
#endif
    pl_scaler_tests(gpu);
    pl_render_tests(gpu);
    pl_custom_mix_tests(gpu);
    peak_record_tests(gpu);
    source_time_tests(gpu);
    source_time_boundary_tests(gpu);
    source_time_field_tests(gpu);
    pl_render_sampling_tests(gpu);
    pl_render_offset_tests(gpu);
    pl_render_bits_tests(gpu);
    pl_render_deband_tests(gpu);
    pl_ycbcr_tests(gpu);

    REQUIRE(!pl_gpu_is_failed(gpu));
}

void gpu_interop_tests(pl_gpu gpu)
{
    pl_test_export_import(gpu, PL_HANDLE_DMA_BUF);
    pl_test_host_ptr(gpu);

    REQUIRE(!pl_gpu_is_failed(gpu));
}
