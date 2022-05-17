/***************
Purpose: Analysis of immigrant shares in population and speech sentiment analysis for congressional records paper 
Author: Harriet Brookes Gray
Date: 
Input: datasheet_05-02-2022.csv
Output: pro_anti_regressions.tex
***************/

/******************
*Set up 
******************/

*set directories 
global congress_project ""

clear 


/******************
*Step 1: clean groups data 
******************/
*import csv file 
import delimited "${congress_project}/data/input/datasheet_05-02-2022.csv"

*collapse 
collapse is_pro is_anti is_neutral frac_total_pop frac_foreign_pop sei, by(decade country continent)

*destring country and continent
encode country, gen(country_factor)
encode continent, gen(continent_factor)

*scale share of foreign born 
gen shr_foreign_pop = frac_foreign_pop*100

*create decade counter
gen decade_counter = 1 if decade == 1880 
replace decade_counter = 2 if decade == 1890
replace decade_counter = 3 if decade == 1900
replace decade_counter = 4 if decade == 1910
replace decade_counter = 5 if decade == 1920
replace decade_counter = 6 if decade == 1930
replace decade_counter = 7 if decade == 1940
replace decade_counter = 8 if decade == 1950
replace decade_counter = 9 if decade == 1960
replace decade_counter = 10 if decade == 1970
replace decade_counter = 11 if decade == 1980
replace decade_counter = 12 if decade == 1990
replace decade_counter = 13 if decade == 2000
replace decade_counter = 14 if decade == 2010
replace decade_counter = 15 if decade == 2020

*calculate outcome variable 
gen pro_anti = is_pro - is_anti


/******************
*Step 2: analysis 
******************/

*By country
reg pro_anti i.decade i.country_factor shr_foreign_pop sei, robust 
estimates store reg4, title(share pro - share anti)

estout reg4 using "${congress_project}/regressions/share_anti_pro_regs.tex", cells(b(star fmt(3)) se(par fmt(2))) legend label varlabels(_cons Constant) stats(r2) replace 

*SEI figure 
twoway (line sei decade if country == "China") (line sei decade if country == "Cuba") (line sei decade if country == "Germany") (line sei decade if country == "Greece") (line sei decade if country == "Haiti") (line sei decade if country == "Hungary") (line sei decade if country == "Ireland") (line sei decade if country == "Italy") (line sei decade if country == "Japan") (line sei decade if country == "Mexico") (line sei decade if country == "Philippines") (line sei decade if country == "Poland") (line sei decade if country == "Russia") (line sei decade if country == "Vietnam"), legend(col(4) label(1 "China") label(2 "Cuba") label(3 "Germany") label(4 "Greece") label(5 "Haiti") label(6 "Hungary") label(7 "Ireland") label(8 "Italy") label(9 "Japan") label(10 "Mexico") label(11 "Philippines") label(12 "Poland") label(13 "Russia") label(14 "Vietnam")) xlabel(1880(10)2020)

graph export "${congress_project}/figures/SEI.pdf", as(pdf) name("Graph") replace





