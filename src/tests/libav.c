#include "utils.h"
#include "libplacebo/utils/libav.h"

#ifdef PL_HAVE_LAV_HDR
static void test_invalid_hdr10plus_metadata(void)
{
    struct pl_hdr_metadata hdr = {
        .scene_avg = 100.0f,
    };
    AVDynamicHDRPlus hdr10plus = {
        .application_version = 1,
        .num_windows = PL_ARRAY_SIZE(hdr10plus.params) + 1,
    };
    pl_map_hdr_metadata(&hdr, &(struct pl_av_hdr_metadata) {
        .dhp = &hdr10plus,
    });
    REQUIRE_CMP(hdr.scene_avg, ==, 100.0f, "f");

    hdr10plus.num_windows = 1;
    hdr10plus.params[0].num_distribution_maxrgb_percentiles =
        PL_ARRAY_SIZE(hdr10plus.params[0].distribution_maxrgb) + 1;
    pl_map_hdr_metadata(&hdr, &(struct pl_av_hdr_metadata) {
        .dhp = &hdr10plus,
    });
    REQUIRE_CMP(hdr.scene_avg, ==, 100.0f, "f");

    hdr10plus.params[0].num_distribution_maxrgb_percentiles = 0;
    hdr10plus.params[0].num_bezier_curve_anchors =
        PL_ARRAY_SIZE(hdr10plus.params[0].bezier_curve_anchors) + 1;
    pl_map_hdr_metadata(&hdr, &(struct pl_av_hdr_metadata) {
        .dhp = &hdr10plus,
    });
    REQUIRE_CMP(hdr.scene_avg, ==, 100.0f, "f");

    hdr10plus.params[0].num_bezier_curve_anchors = 0;
    hdr10plus.params[0].average_maxrgb = (AVRational) {1, 2};
    pl_map_hdr_metadata(&hdr, &(struct pl_av_hdr_metadata) {
        .dhp = &hdr10plus,
    });
    REQUIRE_CMP(hdr.scene_avg, ==, 5000.0f, "f");
}
#endif

#ifdef PL_HAVE_LAV_DOLBY_VISION
static void test_checked_dovi_metadata(void)
{
    size_t metadata_size = 0;
    AVDOVIMetadata *metadata = av_dovi_metadata_alloc(&metadata_size);
    REQUIRE(metadata);

    AVDOVIRpuDataHeader *header = av_dovi_get_header(metadata);
    AVDOVIDataMapping *mapping = av_dovi_get_mapping(metadata);
    AVDOVIColorMetadata *color = av_dovi_get_color(metadata);
    header->bl_bit_depth = 10;
    header->coef_log2_denom = 12;
    header->disable_residual_flag = 1;
    for (int c = 0; c < 3; c++) {
        mapping->curves[c].num_pivots = 2;
        mapping->curves[c].pivots[0] = 0;
        mapping->curves[c].pivots[1] = 1023;
        mapping->curves[c].mapping_idc[0] = AV_DOVI_MAPPING_POLYNOMIAL;
        mapping->curves[c].poly_order[0] = 1;
        color->ycc_to_rgb_offset[c] = (AVRational) {0, 1};
    }
    for (int n = 0; n < 9; n++) {
        color->ycc_to_rgb_matrix[n] =
            (AVRational) {(n % 4) == 0, 1};
        color->rgb_to_lms_matrix[n] =
            (AVRational) {(n % 4) == 0, 1};
    }

    AVFrame *frame = av_frame_alloc();
    REQUIRE(frame);
    frame->format = AV_PIX_FMT_YUV420P10LE;
    AVFrameSideData *sd = av_frame_new_side_data(
        frame, AV_FRAME_DATA_DOVI_METADATA, metadata_size);
    REQUIRE(sd);
    size_t header_offset = metadata->header_offset;
    memcpy(sd->data, metadata, metadata_size);
    av_free(metadata);

    struct pl_color_space color_space = {0};
    struct pl_color_repr repr = {0};
    struct pl_dovi_metadata dovi = {0};
    REQUIRE(pl_map_avframe_dovi_metadata(
        &color_space, &repr, &dovi, frame));
    REQUIRE_CMP(repr.sys, ==, PL_COLOR_SYSTEM_DOLBYVISION, "u");
    REQUIRE_CMP(repr.levels, ==, PL_COLOR_LEVELS_LIMITED, "u");

    uint8_t *aligned_data = sd->data;
    sd->data++;
    REQUIRE(!pl_map_avframe_dovi_metadata(
        &color_space, &repr, &dovi, frame));
    sd->data = aligned_data;

    ((AVDOVIMetadata *) sd->data)->header_offset = metadata_size;
    REQUIRE(!pl_map_avframe_dovi_metadata(
        &color_space, &repr, &dovi, frame));

    AVDOVIMetadata *side_metadata = (AVDOVIMetadata *) sd->data;
    side_metadata->header_offset = header_offset;
    side_metadata->num_ext_blocks = 2;
    side_metadata->ext_block_size++;
    REQUIRE(!pl_map_avframe_dovi_metadata(
        &color_space, &repr, &dovi, frame));
    av_frame_free(&frame);
}
#endif

int main()
{
    struct pl_plane_data data[4] = {0};
    struct pl_bit_encoding bits;

    // Make sure we don't crash on any av pixfmt
    const AVPixFmtDescriptor *desc = NULL;
    while ((desc = av_pix_fmt_desc_next(desc)))
        pl_plane_data_from_pixfmt(data, &bits, av_pix_fmt_desc_get_id(desc));

#define TEST(pixfmt, reference)                                                 \
    do {                                                                        \
        int planes = pl_plane_data_from_pixfmt(data, &bits, pixfmt);            \
        REQUIRE_CMP(planes, ==, sizeof(reference) / sizeof(*reference), "d");   \
        REQUIRE_MEMEQ(data, reference, sizeof(reference));                      \
    } while (0)

    // Planar and semiplanar formats
    static const struct pl_plane_data yuvp8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {1},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {2},
            .pixel_stride = 1,
        }
    };

    TEST(AV_PIX_FMT_YUV420P, yuvp8);
    TEST(AV_PIX_FMT_YUV422P, yuvp8);
    TEST(AV_PIX_FMT_YUV444P, yuvp8);
    TEST(AV_PIX_FMT_YUV410P, yuvp8);
    TEST(AV_PIX_FMT_YUV411P, yuvp8);
    TEST(AV_PIX_FMT_YUV440P, yuvp8);

    static const struct pl_plane_data yuvap8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {1},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {2},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {3},
            .pixel_stride = 1,
        }
    };

    TEST(AV_PIX_FMT_YUVA420P, yuvap8);

    static const struct pl_plane_data yuvp16[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {0},
            .pixel_stride = 2,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {1},
            .pixel_stride = 2,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {2},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_YUV420P10LE, yuvp16);
    TEST(AV_PIX_FMT_YUV420P16LE, yuvp16);

    static const struct pl_plane_data nv12[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8},
            .component_map = {1, 2},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_NV12, nv12);

    static const struct pl_plane_data nv21[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8},
            .component_map = {2, 1},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_NV21, nv21);

    static const struct pl_plane_data p016[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {0},
            .pixel_stride = 2,
        }, {
            .type = PL_FMT_UNORM,
            .component_size = {16, 16},
            .component_map = {1, 2},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_P010LE, p016);
    TEST(AV_PIX_FMT_P016LE, p016);

    // Packed formats
    static const struct pl_plane_data r8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8},
            .component_map = {0},
            .pixel_stride = 1,
        }
    };

    TEST(AV_PIX_FMT_GRAY8, r8);

    static const struct pl_plane_data rg8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8},
            .component_map = {0, 1},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_GRAY8A, rg8);

    static const struct pl_plane_data rgb8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {0, 1, 2},
            .pixel_stride = 3,
        }
    };

    TEST(AV_PIX_FMT_RGB24, rgb8);

    static const struct pl_plane_data bgr8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {2, 1, 0},
            .pixel_stride = 3,
        }
    };

    TEST(AV_PIX_FMT_BGR24, bgr8);

    static const struct pl_plane_data rgbx8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {0, 1, 2},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_RGB0, rgbx8);

    static const struct pl_plane_data xrgb8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8},
            .component_map = {0, 1, 2},
            .component_pad = {8, 0, 0},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_0RGB, xrgb8);

    static const struct pl_plane_data rgba8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {0, 1, 2, 3},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_RGBA, rgba8);

    static const struct pl_plane_data argb8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {3, 0, 1, 2},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_ARGB, argb8);

    static const struct pl_plane_data bgra8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {2, 1, 0, 3},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_BGRA, bgra8);

    static const struct pl_plane_data abgr8[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {8, 8, 8, 8},
            .component_map = {3, 2, 1, 0},
            .pixel_stride = 4,
        }
    };

    TEST(AV_PIX_FMT_ABGR, abgr8);

    static const struct pl_plane_data r16[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16},
            .component_map = {0},
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_GRAY16LE, r16);

    static const struct pl_plane_data rgb16[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16, 16, 16},
            .component_map = {0, 1, 2},
            .pixel_stride = 6,
        }
    };

    TEST(AV_PIX_FMT_RGB48LE, rgb16);

    static const struct pl_plane_data rgb16be[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16, 16, 16},
            .component_map = {0, 1, 2},
            .pixel_stride = 6,
            .swapped = true,
        }
    };

    TEST(AV_PIX_FMT_RGB48BE, rgb16be);

    static const struct pl_plane_data rgba16[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16, 16, 16, 16},
            .component_map = {0, 1, 2, 3},
            .pixel_stride = 8,
        }
    };

    TEST(AV_PIX_FMT_RGBA64LE, rgba16);

    static const struct pl_plane_data rgba16be[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {16, 16, 16, 16},
            .component_map = {0, 1, 2, 3},
            .pixel_stride = 8,
            .swapped = true,
        }
    };

    TEST(AV_PIX_FMT_RGBA64BE, rgba16be);

    static const struct pl_plane_data rgb565[] = {
        {
            .type = PL_FMT_UNORM,
            .component_size = {5, 6, 5},
            .component_map = {2, 1, 0}, // LSB to MSB
            .pixel_stride = 2,
        }
    };

    TEST(AV_PIX_FMT_RGB565LE, rgb565);

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 37, 100)

    static const struct pl_plane_data rgb32f[] = {
        {
            .type = PL_FMT_FLOAT,
            .component_size = {32, 32, 32},
            .component_map = {0, 1, 2},
            .pixel_stride = 12,
        }
    };

    TEST(AV_PIX_FMT_RGBF32LE, rgb32f);

#endif

    // Test pl_frame <- AVFrame bridge
    struct pl_frame image;
    AVFrame *frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_RGBA;
    pl_frame_from_avframe(&image, frame);
    REQUIRE_CMP(image.num_planes, ==, 1, "d");
    REQUIRE_CMP(image.repr.sys, ==, PL_COLOR_SYSTEM_RGB, "u");

    // Test inverse mapping
    struct pl_color_space csp = image.color;
    pl_color_space_infer(&csp);
    pl_avframe_set_color(frame, csp);
    pl_avframe_set_repr(frame, image.repr);
    pl_avframe_set_profile(frame, image.profile);
    pl_frame_from_avframe(&image, frame);
    pl_color_space_infer(&image.color);
    REQUIRE(pl_color_space_equal(&csp, &image.color));
    av_frame_free(&frame);

#ifdef PL_HAVE_LAV_HDR
    test_invalid_hdr10plus_metadata();
#endif
#ifdef PL_HAVE_LAV_DOLBY_VISION
    test_checked_dovi_metadata();
#endif

    // Test enum functions
    for (enum pl_color_system sys = 0; sys < PL_COLOR_SYSTEM_COUNT; sys++) {
        enum AVColorSpace spc = pl_system_to_av(sys);
        enum pl_color_system sys2 = pl_system_from_av(spc);
        // Exception to the rule, due to different handling in libav*
        if (sys2 && sys != PL_COLOR_SYSTEM_BT_2100_HLG)
            REQUIRE_CMP(sys, ==, sys2, "u");
    }

    for (enum pl_color_levels lev = 0; lev < PL_COLOR_LEVELS_COUNT; lev++) {
        enum AVColorRange range = pl_levels_to_av(lev);
        enum pl_color_levels lev2 = pl_levels_from_av(range);
        REQUIRE_CMP(lev, ==, lev2, "u");
    }

    for (enum pl_color_primaries prim = 0; prim < PL_COLOR_PRIM_COUNT; prim++) {
        enum AVColorPrimaries avpri = pl_primaries_to_av(prim);
        enum pl_color_primaries prim2 = pl_primaries_from_av(avpri);
        if (prim2)
            REQUIRE_CMP(prim, ==, prim2, "u");
    }

    for (enum pl_color_transfer trc = 0; trc < PL_COLOR_TRC_COUNT; trc++) {
        enum AVColorTransferCharacteristic avtrc = pl_transfer_to_av(trc);
        enum pl_color_transfer trc2 = pl_transfer_from_av(avtrc);
        if (trc2)
            REQUIRE_CMP(trc, ==, trc2, "u");
    }

    for (enum pl_chroma_location loc = 0; loc < PL_CHROMA_COUNT; loc++) {
        enum AVChromaLocation avloc = pl_chroma_to_av(loc);
        enum pl_chroma_location loc2 = pl_chroma_from_av(avloc);
        REQUIRE_CMP(loc, ==, loc2, "u");
    }
}
