from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)


def export_pdf(
        question,
        answer,
        filename
):

    doc = SimpleDocTemplate(
        filename
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            f"<b>Question:</b> {question}",
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1,12)
    )

    story.append(
        Paragraph(
            answer,
            styles["Normal"]
        )
    )

    doc.build(story)