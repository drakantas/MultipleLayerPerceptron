class App {
    constructor()
    {
        this.handleClickCell();
    }

    toggleCell(cell)
    {
        if (cell.hasClass('checked'))
            cell.removeClass('checked');
        else
            cell.addClass('checked');

        this.toggleCellCheckbox(cell);
    }

    handleClickCell()
    {
        const cells = $('.character-builder > tbody > tr > td');
        let $this = this;

        $(cells).on('click', function (e) {
            const cell = $(e.currentTarget);

            $this.toggleCell(cell);
        });
    }

    toggleCellCheckbox(cell)
    {
        const checkbox = $(cell.find('input[type="checkbox"]')[0]);

        if (checkbox.is(':checked'))
            checkbox.prop('checked', false);
        else
            checkbox.prop('checked', true);
    }
}

let app = new App();
