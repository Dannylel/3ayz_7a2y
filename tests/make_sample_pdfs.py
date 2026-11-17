"""Generate two SYNTHETIC Arabic 'law' PDFs (invented text, not real legislation)
to smoke-test the pipeline. Rendered with reportlab in visual order + presentation
forms, which is the awkward case many real Arabic PDFs fall into."""
import sys
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))

PUB = "ينشر هذا القانون في الجريدة الرسمية، ويعمل به من اليوم التالي لتاريخ نشره، ويبصم بخاتم الدولة، وينفذ كقانون من قوانينها."

LAW_A = ("قانون تجريبي رقم 1 لسنة 2020 بشأن إيجار الأماكن", [
    "يسري هذا القانون على الأماكن المؤجرة لأغراض السكنى أو لغير ذلك من الأغراض، سواء كانت مفروشة أو غير مفروشة، في جميع أنحاء الجمهورية.",
    "يجب أن يكون عقد الإيجار مكتوباً، وأن يتضمن بيان العين المؤجرة ومدة الإيجار ومقدار الأجرة وطريقة سدادها. " * 6,
    "يلتزم المؤجر بتسليم العين المؤجرة وملحقاتها في حالة تصلح معها لأن تفي بما أعدت له من المنفعة، ويلتزم بإجراء الترميمات الضرورية طوال مدة العقد.",
    "يلتزم المستأجر بسداد الأجرة في المواعيد المتفق عليها، فإذا تأخر عن السداد مدة تزيد على ثلاثة أشهر جاز للمؤجر أن يطلب فسخ العقد وإخلاء العين.",
    "لا يجوز للمستأجر التنازل عن الإيجار أو التأجير من الباطن إلا بإذن كتابي صريح من المالك.",
    "تزاد الأجرة سنوياً بنسبة لا تجاوز سبعة في المائة من قيمة آخر أجرة مستحقة.",
    "ينتهي عقد الإيجار بانتهاء المدة المتفق عليها، ما لم يتفق الطرفان على تجديده كتابة.",
    "يعاقب بغرامة لا تقل عن ألف جنيه كل من خالف أحكام المادة 5 من هذا القانون.",
    PUB,
])

LAW_B = ("قانون تجريبي رقم 2 لسنة 2021 بشأن علاقات العمل", [
    "تسري أحكام هذا القانون على العاملين في القطاع الخاص، ولا تسري على العاملين بأجهزة الدولة.",
    "يلتزم صاحب العمل بتحرير عقد العمل كتابة من ثلاث نسخ، يسلم العامل نسخة منها، وأن يتضمن بيانات الأجر ونوع العمل ومدته.",
    "لا يجوز أن تزيد مدة الاختبار على ثلاثة أشهر، ولا يجوز تعيين العامل تحت الاختبار أكثر من مرة لدى صاحب عمل واحد. " * 5,
    "تحدد ساعات العمل بثماني ساعات يومياً بحد أقصى، ولا تدخل فيها الفترات المخصصة لتناول الطعام والراحة.",
    "يستحق العامل إجازة سنوية مدفوعة الأجر مدتها واحد وعشرون يوماً، تزاد إلى ثلاثين يوماً لمن أمضى عشر سنوات في الخدمة.",
    "يجوز لصاحب العمل إنهاء عقد العمل غير محدد المدة إذا أخل العامل بالتزاماته الجوهرية، وذلك بعد إخطاره كتابة قبل الإنهاء بثلاثة أشهر.",
    "يلتزم المستأجر لسكن العمال بسداد الأجرة في موعدها وفقاً لما يحدده صاحب العمل.",
    PUB,
])


def shape(s):
    return get_display(arabic_reshaper.reshape(s))


def wrap(text, width_chars=70):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def make(path, law):
    title, arts = law
    c = canvas.Canvas(str(path), pagesize=A4)
    W, H = A4
    y = H - 60

    def line(s, size=12):
        nonlocal y
        if y < 60:
            c.showPage()
            y = H - 60
        c.setFont("DejaVu", size)
        c.drawRightString(W - 50, y, shape(s))
        y -= size + 8

    line(title, 15)
    line("")
    for i, a in enumerate(arts, 1):
        line(f"مادة ({i})", 13)
        for l in wrap(a):
            line(l)
        line("")
    c.save()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_pdfs")
    out.mkdir(parents=True, exist_ok=True)
    make(out / "law_rent_synthetic.pdf", LAW_A)
    make(out / "law_labor_synthetic.pdf", LAW_B)
    print("wrote", list(out.iterdir()))
