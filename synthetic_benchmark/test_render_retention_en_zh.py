from PIL import Image, ImageDraw, ImageFont

from render_retention_en_zh import (
    FontFace,
    doc_split,
    make_plan,
    near_duplicate,
    normalize_target,
    parquet_row,
    stable_int,
    wrap_chinese,
    wrap_english,
)


def test_target_is_nfc_with_clean_line_ends():
    assert normalize_target("e\u0301  \r\nline  \n") == "é\nline"


def test_near_duplicate_detects_normalized_copy():
    candidate = "A sufficiently long sentence that is copied exactly."
    reference = "asufficientlylongsentencethatiscopiedexactly"
    assert near_duplicate(candidate, [reference]) == (True, 1.0)


def test_stable_seed_fits_arrow_int64():
    assert 0 <= stable_int("retention", 1) < 2**63
    assert parquet_row({"document_augmentation_seed": 2**64 - 1}) == {
        "document_augmentation_seed": str(2**64 - 1)
    }


def test_english_wrap_does_not_hyphenate():
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    draw = ImageDraw.Draw(Image.new("L", (500, 100)))
    lines = wrap_english("The quick brown fox jumps over the lazy dog.", draw, font, 200)
    assert " ".join(lines) == "The quick brown fox jumps over the lazy dog."
    assert all(not line.endswith("-") for line in lines)


def test_chinese_wrap_preserves_every_character():
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    draw = ImageDraw.Draw(Image.new("L", (500, 100)))
    text = "天地玄黃，宇宙洪荒。日月盈昃，辰宿列張。"
    assert "".join(wrap_chinese(text, draw, font, 200)) == text


def test_train_only_continuation_ids():
    def train_doc(language):
        pageid = 1
        while True:
            doc = {
                "pageid": str(pageid),
                "source": f"{language}.wikipedia.org",
                "title": "Continuation source",
                "url": f"https://{language}.wikipedia.org/?curid={pageid}",
                "license": "CC BY-SA 4.0",
                "variant": language,
                "text": ("A sufficiently long paragraph for deterministic continuation rendering. " * 20),
            }
            if doc_split(doc) == "train":
                return doc
            pageid += 1

    face = FontFace(
        "DejaVu Sans:style=Book",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVu Sans",
        "Book",
        0,
        "Bitstream Vera license",
    )
    fonts = {
        language: {split: [face] for split in ("train", "val", "test")}
        for language in ("en", "zh")
    }
    rows, _ = make_plan(
        {"en": [train_doc("en")], "zh": [train_doc("zh")]},
        fonts,
        {"en": [], "zh": []},
        2,
        start_image_id=5001,
        train_only=True,
    )
    assert {row["suggested_split"] for row in rows} == {"train"}
    assert [row["image_id"] for row in rows if row["language_group"] == "en"] == [5001, 5002]
    assert [row["image_id"] for row in rows if row["language_group"] == "zh"] == [5001, 5002]
