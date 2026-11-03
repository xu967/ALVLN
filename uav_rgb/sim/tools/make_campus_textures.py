#!/usr/bin/env python3
"""生成项目自有的低成本校园材质；固定种子，不下载外部资产。"""
from pathlib import Path
import random
from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "worlds" / "campus_textures"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for index, (name, base) in enumerate((("pavement", (118, 119, 115)),
                                          ("brick", (153, 112, 86)),
                                          ("bark", (91, 67, 42)))):
        rng = random.Random(1200 + index)
        image = Image.new("RGB", (128, 128))
        pixels = image.load()
        for y in range(128):
            for x in range(128):
                noise = rng.randint(-10, 10)
                if name == "pavement" and (x % 32 < 2 or y % 32 < 2):
                    noise -= 30
                if name == "brick" and (y % 16 < 2 or (x + (y // 16 % 2) * 16) % 32 < 2):
                    noise = 35
                if name == "bark" and x % 13 < 2:
                    noise -= 28
                pixels[x, y] = tuple(max(0, min(255, c + noise)) for c in base)
        image.save(ASSETS / f"{name}.png")


if __name__ == "__main__":
    main()
