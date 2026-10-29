import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import bridge_server_api as api


class CharacterImageTests(unittest.TestCase):
    def request_image(self, image, **save_options):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "character.png"
            image.save(image_path, **save_options)
            with patch.object(api.state, "chars", [{"image_path": str(image_path)}]):
                response = TestClient(api.app).get("/characters/0/image")
        self.assertEqual(response.status_code, 200)
        return response, Image.open(io.BytesIO(response.content))

    def test_rgba_png_preserves_transparent_and_semitransparent_pixels(self):
        image = Image.new("RGBA", (640, 640), (255, 0, 255, 0))
        image.paste((20, 180, 60, 128), (200, 200, 400, 400))
        response, result = self.request_image(image)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(result.size, (640, 640))
        self.assertEqual(result.convert("RGBA").getpixel((10, 10))[3], 0)
        self.assertEqual(result.convert("RGBA").getpixel((300, 300)), (20, 180, 60, 128))

    def test_palette_png_preserves_transparency(self):
        image = Image.new("P", (640, 640), 0)
        image.putpalette([255, 0, 255, 20, 180, 60] + [0] * 762)
        image.paste(1, (200, 200, 400, 400))
        response, result = self.request_image(image, transparency=0)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(result.convert("RGBA").getpixel((10, 10))[3], 0)
        self.assertEqual(result.convert("RGBA").getpixel((300, 300)), (20, 180, 60, 255))

    def test_resized_edges_do_not_show_hidden_transparent_rgb(self):
        image = Image.new("RGBA", (1280, 1280), (255, 0, 255, 0))
        image.paste((0, 255, 0, 255), (401, 401, 879, 879))
        _, result = self.request_image(image)
        rgba = result.convert("RGBA")
        pixels = [rgba.getpixel((x, 320)) for x in range(rgba.width)]
        self.assertTrue(any(0 < alpha < 255 for _, _, _, alpha in pixels))
        self.assertTrue(all((red, green, blue) == (0, 255, 0)
                            for red, green, blue, alpha in pixels if alpha > 0))

    def test_grayscale_alpha_and_rgb_transparency_key(self):
        for image, options in (
            (Image.new("LA", (640, 640), (180, 64)), {}),
            (Image.new("RGB", (640, 640), (255, 0, 255)), {"transparency": (255, 0, 255)}),
        ):
            with self.subTest(mode=image.mode):
                _, result = self.request_image(image, **options)
                expected_alpha = 64 if image.mode == "LA" else 0
                self.assertEqual(result.convert("RGBA").getpixel((300, 300))[3], expected_alpha)

    def test_opaque_image_retains_jpeg_and_top_center_crop(self):
        image = Image.new("RGB", (800, 640), (30, 150, 70))
        image.paste((255, 0, 0), (0, 0, 80, 640))
        image.paste((0, 0, 255), (720, 0, 800, 640))
        response, result = self.request_image(image)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(result.size, (640, 640))
        for point in ((0, 0), (320, 320), (639, 639)):
            self.assertTrue(all(abs(a - b) <= 2 for a, b in zip(result.getpixel(point), (30, 150, 70))))
