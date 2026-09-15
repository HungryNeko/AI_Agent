import pytest

from tools import fileReader
from tools.fileReader import FileReadRequest
from tools.settings import FileReaderSettings


def settings(tmp_path, *, max_output_chars=60_000):
    return FileReaderSettings(
        mode="auto",
        root=str(tmp_path),
        max_file_bytes=25_000_000,
        max_output_chars=max_output_chars,
    )


def test_reads_utf8_text_without_extension_whitelist_by_category(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Knowledge\n\nA durable fact.", encoding="utf-8")

    result = fileReader.read(FileReadRequest(path="notes.md"), settings(tmp_path))

    assert result["format"] == "md"
    assert result["content"] == "# Knowledge\n\nA durable fact."
    assert result["truncated"] is False


def test_reads_pdf_pages_and_reports_image_only_pdf(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "scan.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    writer.write(path)
    result = fileReader.read(
        FileReadRequest(path="scan.pdf", start_page=2, end_page=2),
        settings(tmp_path),
    )

    assert result["pageCount"] == 2
    assert result["startPage"] == 2
    assert result["endPage"] == 2
    assert result["needsOcr"] is True
    assert "[Page 2]" in result["content"]


def test_reads_docx_paragraphs_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    document = docx.Document()
    document.add_paragraph("Quarterly report")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Name"
    table.rows[0].cells[1].text = "Value"
    document.save(path)

    result = fileReader.read(FileReadRequest(path="report.docx"), settings(tmp_path))

    assert "Quarterly report" in result["content"]
    assert "Name\tValue" in result["content"]
    assert result["tableCount"] == 1


def test_reads_pptx_slide_text(tmp_path):
    pptx = pytest.importorskip("pptx")
    path = tmp_path / "deck.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Roadmap"
    presentation.save(path)

    result = fileReader.read(FileReadRequest(path="deck.pptx"), settings(tmp_path))

    assert result["slideCount"] == 1
    assert "[Slide 1]" in result["content"]
    assert "Roadmap" in result["content"]


def test_reads_selected_xlsx_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "facts.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Facts"
    workbook.active.append(["key", "value"])
    workbook.active.append(["language", "Chinese"])
    workbook.active.append(["zero", 0])
    workbook.create_sheet("Ignored").append(["skip"])
    workbook.save(path)

    result = fileReader.read(
        FileReadRequest(path="facts.xlsx", sheet="Facts"),
        settings(tmp_path),
    )

    assert result["sheetsRead"] == ["Facts"]
    assert "key\tvalue" in result["content"]
    assert "language\tChinese" in result["content"]
    assert "zero\t0" in result["content"]
    assert "skip" not in result["content"]


def test_local_upload_url_maps_to_upload_directory(tmp_path):
    path = tmp_path / "backend" / "runtime" / "uploads" / "abc" / "notes.txt"
    path.parent.mkdir(parents=True)
    path.write_text("uploaded content", encoding="utf-8")

    result = fileReader.read(
        FileReadRequest(path="/api/uploads/abc/notes.txt"),
        settings(tmp_path),
    )

    assert result["path"] == "backend/runtime/uploads/abc/notes.txt"
    assert result["content"] == "uploaded content"


def test_blocks_secret_and_non_upload_runtime_files(tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=secret", encoding="utf-8")
    conversation = tmp_path / "backend" / "runtime" / "conversations" / "one.json"
    conversation.parent.mkdir(parents=True)
    conversation.write_text('{"secret":"value"}', encoding="utf-8")

    with pytest.raises(ValueError, match="secret-bearing"):
        fileReader.read(FileReadRequest(path=".env"), settings(tmp_path))
    with pytest.raises(ValueError, match="only uploaded files"):
        fileReader.read(
            FileReadRequest(path="backend/runtime/conversations/one.json"),
            settings(tmp_path),
        )


def test_rejects_legacy_word_format_with_conversion_hint(tmp_path):
    path = tmp_path / "old.doc"
    path.write_bytes(b"legacy")

    with pytest.raises(ValueError, match=r"convert the file to \.docx"):
        fileReader.read(FileReadRequest(path="old.doc"), settings(tmp_path))


def test_rejects_zero_based_page_numbers(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "one.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(path)

    with pytest.raises(ValueError, match="positive 1-based"):
        fileReader.read(FileReadRequest(path="one.pdf", start_page=0), settings(tmp_path))
