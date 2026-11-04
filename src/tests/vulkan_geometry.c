#include "utils.h"
#include "vulkan/swapchain_geometry.h"
#include <libplacebo/renderer.h>

static VkSurfaceCapabilitiesKHR capabilities(void)
{
    return (VkSurfaceCapabilitiesKHR) {
        .currentExtent = {2400, 1080},
        .minImageExtent = {1, 1},
        .maxImageExtent = {4096, 4096},
        .currentTransform = VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR,
        .supportedTransforms = 511,
    };
}

static void check_geometry(void)
{
    const int sizes[][2] = {{2400, 1080}, {1080, 2400}, {127, 95}, {97, 97}};
    const VkSurfaceTransformFlagBitsKHR transforms[] = {
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR,
        VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR,
        VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR,
    };
    for (int t = 0; t < 4; t++) {
        for (int n = 0; n < PL_ARRAY_SIZE(sizes); n++) {
            VkSurfaceCapabilitiesKHR caps = capabilities();
            caps.currentTransform = transforms[t];
            int w = sizes[n][0], h = sizes[n][1];
            struct vk_swapchain_geometry g;
            REQUIRE(vk_swapchain_geometry(&caps, true, w, h, 0, 0, &g));
            REQUIRE_CMP(g.transform, ==, transforms[t], "u");
            int inverse = (4 - t) % 4;
            REQUIRE_CMP(g.rotation, ==, inverse, "d");
            REQUIRE_CMP(g.width, ==, w, "d");
            REQUIRE_CMP(g.height, ==, h, "d");
            unsigned storage_w = t % 2 ? h : w;
            unsigned storage_h = t % 2 ? w : h;
            REQUIRE_CMP(g.extent.width, ==, storage_w, "u");
            REQUIRE_CMP(g.extent.height, ==, storage_h, "u");

            // Android applies the inverse preTransform after rendering. The
            // renderer applies source - target, preserving source orientation.
            for (int source = 0; source < 4; source++) {
                int rendered = pl_rotation_normalize(source - g.rotation);
                REQUIRE_CMP(pl_rotation_normalize(rendered - t), ==, source, "d");
            }

            // Unknown surface dimensions must reuse logical, not storage size.
            caps.currentExtent = (VkExtent2D) {UINT32_MAX, UINT32_MAX};
            for (int i = 0; i < 3; i++) {
                REQUIRE(vk_swapchain_geometry(&caps, true, 0, 0, g.width, g.height, &g));
                REQUIRE_CMP(g.width, ==, w, "d");
                REQUIRE_CMP(g.height, ==, h, "d");
            }
            REQUIRE(vk_swapchain_geometry(&caps, true, w + 2, 0, w, h, &g));
            REQUIRE_CMP(g.width, ==, w + 2, "d");
            REQUIRE_CMP(g.height, ==, h, "d");

            // Default-off keeps the original dimensions and zero rotation.
            REQUIRE(vk_swapchain_geometry(&caps, false, w, h, 0, 0, &g));
            REQUIRE_CMP(g.rotation, ==, 0, "d");
            REQUIRE_CMP(g.transform, ==, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, "u");
            REQUIRE_CMP(g.extent.width, ==, w, "u");
            REQUIRE_CMP(g.extent.height, ==, h, "u");
        }
    }

    VkSurfaceCapabilitiesKHR caps = capabilities();
    struct vk_swapchain_geometry g;
    REQUIRE(vk_swapchain_geometry(&caps, true, 0, 0, 0, 0, &g));
    REQUIRE_CMP(g.extent.width, ==, 1080, "u");
    REQUIRE_CMP(g.extent.height, ==, 2400, "u");
    caps.maxImageExtent = (VkExtent2D) {1000, 2000};
    REQUIRE(vk_swapchain_geometry(&caps, true, 2400, 1080, 0, 0, &g));
    REQUIRE_CMP(g.width, ==, 2000, "d");
    REQUIRE_CMP(g.height, ==, 1000, "d");
    struct vk_swapchain_geometry previous = g;
    caps.currentTransform = VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR;
    REQUIRE(!vk_swapchain_geometry(&caps, true, 100, 100, 0, 0, &g));
    REQUIRE(!memcmp(&g, &previous, sizeof(g)));
    caps.currentTransform = VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR;
    REQUIRE(!vk_swapchain_geometry(&caps, true, 100, 100, 0, 0, &g));
    caps.supportedTransforms = VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR;
    REQUIRE(vk_swapchain_geometry(&caps, false, 100, 100, 0, 0, &g));
    REQUIRE_CMP(g.transform, ==, VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR, "u");
    REQUIRE_CMP(g.rotation, ==, 0, "d");
    caps.currentTransform = VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR;
    REQUIRE(!vk_swapchain_geometry(&caps, true, 100, 100, 0, 0, &g));
    caps = capabilities();
    caps.currentExtent = (VkExtent2D) {UINT32_MAX, UINT32_MAX};
    REQUIRE(!vk_swapchain_geometry(&caps, true, 0, 0, 0, 0, &g));
    REQUIRE(!vk_swapchain_geometry(&caps, true, -1, 100, 0, 0, &g));
    caps.minImageExtent = (VkExtent2D) {0, 0};
    REQUIRE(vk_swapchain_geometry(&caps, false, 100, 100, 0, 0, &g));
    caps.minImageExtent.width = caps.maxImageExtent.width + 1;
    REQUIRE(!vk_swapchain_geometry(&caps, true, 100, 100, 0, 0, &g));
}

static void check_target(void)
{
    const struct pl_fmt_t format = {.num_components = 4};
    const struct pl_tex_t texture = {.params = {.w = 95, .h = 127, .format = &format}};
    for (int r = 0; r < 4; r++) {
        struct pl_swapchain_frame swap = {.fbo = &texture, .rotation = r};
        struct pl_frame target;
        pl_frame_from_swapchain(&target, &swap);
        REQUIRE_CMP(target.rotation, ==, r, "d");
        REQUIRE_CMP(target.crop.x1, ==, 95.0, "f");
        REQUIRE_CMP(target.crop.y1, ==, 127.0, "f");
        REQUIRE(target.planes[0].texture == &texture);
    }
}

int main(void)
{
    check_geometry();
    check_target();
    return 0;
}
