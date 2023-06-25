# Version 1.0
import tabula

def pdf_2_table(input_path, output_path):
    tables = tabula.read_pdf(input_path, pages='all', multiple_tables=True)

    html = "<html><body>"
    for table in tables:
        html += "<table>"
        for row in table.values:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</table>"
    html += "</body></html>"

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(html)


pdf_2_table('input.pdf', 'output.html')
