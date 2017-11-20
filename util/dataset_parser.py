from openpyxl import load_workbook
from pathlib import Path


class Parser:
    @staticmethod
    def load(path: Path, sheet: str, cpr: int = 9, cpf: int = 7, fonts: int = 6):
        workbook = load_workbook(str(path.resolve()))
        dataset = list()
        _rows = list()
        _row_counter = 0

        for row in workbook[sheet].rows:
            if _row_counter == 54:
                break

            _rows.append(list())  # Add new row to the _dataset
            for cell in row:
                if len(_rows[-1]) != cpr:
                    _rows[-1].append(cell.value)
                else:
                    _rows.append(list())
                    _rows[-1].append(cell.value)

            _row_counter += 1

        if not dataset:
            for font in range(0, fonts):
                dataset.append(list())

        assert len(_rows) % cpf == 0

        font_index = 0

        for ri, row in enumerate(_rows):
            dataset[font_index].append(row)

            if (ri + 1) % 63 == 0:
                font_index += 1

                if font_index == fonts:
                    font_index = 0

        del font_index, _row_counter, _rows, workbook
