import opendataloader_pdf
# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["../paper/Ao_Open-World_Amodal_Appearance_Completion_CVPR_2025_paper.pdf"],
    output_dir="../ref/",
    format="json,markdown",
)