all:	pdf

pdf:	report_hcs.tex references.bib
	pdflatex report_hcs.tex

cleanup:
	rm report_hcs.log report_hcs.aux

clean:
	rm report_hcs.pdf; make cleanup
