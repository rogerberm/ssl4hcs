all:	pdf

pdf:	report_cassava.tex references.bib
	pdflatex report_cassava.tex && bibtex report_cassava -terse && pdflatex report_cassava.tex; make showastext
	scp report_cassava.pdf /dfs/beroger/homepage/download/cassava/;

showastext:
	pdftotext -layout report_cassava.pdf report_cassava.txt; awk '{print "\t\t\t\t\t\t\t\t\t\t\t" $$0}' report_cassava.txt| LESSCHARSET=utf-8 less

cleanup:
	rm report_cassava.log report_cassava.aux

clean:
	rm report_cassava.pdf; make cleanup
