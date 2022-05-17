/***************
Purpose: Gallup survey on immigration and speech sentiment analysis for congressional records paper 
Author: Harriet Brookes Gray
Date: 
Input: gallup_combined.csv, speech_data.csv
Output: gallup_speech_year_state.dta, anti_decreased_ys_yearfe_with_labels.pdf
***************/


/******************
*Set up 
******************/

*set directories 
global congress_project ""

/******************
*Step 1: clean Gallup data 
******************/
clear all 
import delimited "${congress_project}/data/input/gallup_combined.csv"

*drop 1964 and 1965 since we are not using these years in our analysis 
drop if year == 1964 | year == 1965

*add labels for state 
#delimit ;
label define state 
11 "ME"
12 "NH"
13 "VT"
14 "MA"
15 "RI"
16 "CT"
21 "NY"
22 "NJ"
23 "PA"
24 "MD"
25 "DE"
26 "WV"
27 "DC"
31 "OH"
32 "MI"
33 "IN"
34 "IL"
41 "WI"
42 "MN"
43 "IA"
44 "MO"
45 "ND"
46 "SD"
47 "NE"
48 "KS"
51 "VA"
52 "NC"
53 "SC"
54 "GA"
55 "FL"
56 "KY"
57 "TN"
58 "AL"
59 "MS"
61 "AR"
62 "LA"
63 "OK"
64 "TX"
71 "MT"
72 "AZ"
73 "CO"
74 "ID"
75 "WY"
76 "UT"
77 "NV"
78 "NM"
81 "CA"
82 "OR"
83 "WA"
84 "HI"
85 "AK";
	;
	#delimit cr
	
label values state state 

*label imm_response 
label define imm_response 1 "present" 2 "increased" 3 "decreased" 4 "don't know" 5 "refused"
label values imm_response imm_response 


* Republican is 1 and democrat 5
drop if party == 8 | party == 9 // from codebook, seem to be missing vaalues 

*immigration response variables 
drop if imm_response == 4 | imm_response == 5 

gen imm_decrease = 1 if imm_response == 3 
replace imm_decrease = 0 if imm_decrease == . 

gen imm_increase = 1 if imm_response == 2
replace imm_increase = 0 if imm_increase == . 

gen imm_present = 1 if imm_response == 1 
replace imm_present = 0 if imm_present == . 


*Collapse data to the year-state level and create file to merge to speech data later 
collapse imm_decrease imm_present imm_increase, by(year state) 
tempfile gallup_by_year_state
save `gallup_by_year_state'

/******************
*Step 2: Clean speech data 
******************/
clear all 
import delimited "${congress_project}/data/input/speech_data.csv"

*get year from date 
tostring date, replace
gen year =substr(date, 1, 4)
destring year, replace 

*create variables for speech tone 
gen anti = 1 if tone_label_int == 0 
replace anti = 0 if anti == . 

gen neutral = 1 if tone_label_int == 1
replace neutral = 0 if neutral == . 

gen pro = 1 if tone_label_int == 2 
replace pro = 0 if pro == . 

*create numeric variable for state to merge with gallup data 
gen state_v2 = . 
replace state_v2 = 11 if state == "ME"
replace state_v2 = 12 if state =="NH"
replace state_v2 = 13 if state =="VT"
replace state_v2 = 14 if state =="MA"
replace state_v2 = 15 if state =="RI"
replace state_v2 = 16 if state =="CT"
replace state_v2 = 21 if state =="NY"
replace state_v2 = 22 if state =="NJ"
replace state_v2 = 23 if state =="PA"
replace state_v2 = 24 if state =="MD"
replace state_v2 = 25 if state =="DE"
replace state_v2 = 26 if state =="WV"
replace state_v2 = 27 if state =="DC"
replace state_v2 = 31 if state =="OH"
replace state_v2 = 32 if state =="MI"
replace state_v2 = 33 if state =="IN"
replace state_v2 = 34 if state =="IL"
replace state_v2 = 41 if state =="WI"
replace state_v2 = 42 if state =="MN"
replace state_v2 = 43 if state =="IA"
replace state_v2 = 44 if state =="MO"
replace state_v2 = 45 if state =="ND"
replace state_v2 = 46 if state =="SD"
replace state_v2 = 47 if state =="NE"
replace state_v2 = 48 if state =="KS"
replace state_v2 = 51 if state =="VA"
replace state_v2 = 52 if state =="NC"
replace state_v2 = 53 if state =="SC"
replace state_v2 = 54 if state =="GA"
replace state_v2 = 55 if state =="FL"
replace state_v2 = 56 if state =="KY"
replace state_v2 = 57 if state =="TN"
replace state_v2 = 58 if state =="AL"
replace state_v2 = 59 if state =="MS"
replace state_v2 = 61 if state == "AR"
replace state_v2 = 62 if state == "LA"
replace state_v2 = 63 if state =="OK"
replace state_v2 = 64 if state =="TX"
replace state_v2 = 71 if state =="MT"
replace state_v2 = 72 if state =="AZ"
replace state_v2 = 73 if state =="CO"
replace state_v2 = 74 if state =="ID"
replace state_v2 = 75 if state =="WY"
replace state_v2 = 76 if state =="UT"
replace state_v2 = 77 if state =="NV"
replace state_v2 = 78 if state =="NM"
replace state_v2 = 81 if state =="CA"
replace state_v2 = 82 if state =="OR"
replace state_v2 = 83 if state =="WA"
replace state_v2 = 84 if state =="HI"
replace state_v2 = 85 if state =="AK"

*create weights = number of speeches per year per state
bysort state_v2 year: gen weight = _N 

*collapse state to the year-state level 
collapse anti neutral pro weight, by(year state_v2) 

*rename state 
rename state_v2 state 

*merge with gallup 
merge 1:1 year state using `gallup_by_year_state', keep(3) nogen 

save "${congress_project}/data/output/gallup_speech_year_state.dta", replace 


/******************
*Step 3: Analysis 
******************/
*YEAR FE regression 
reg anti imm_decrease i.year [w=weight], robust 

*now plot the regression 
cap drop y_res x_res
reg anti i.year [w=weight] //get demeaned y
predict y_res, res
reg imm_decrease i.year [w=weight] //get demeaned x 
predict x_res, res
twoway (scatter y_res x_res [w=weight], msymbol(circle_hollow)) (lfit y_res x_res [w=weight], /*text(0.6 0.39 "Slope = 0.27", color(lablue)) text(0.55 0.45 "Standard error =.078")*/ lcolor(black) legend(off)), ytitle("y (share of anti-immigration speeches) residuals") xtitle("x (share of decreased immigration answers) residuals") //uncomment text to include regression slope and SE on graph 

graph export "${congress_project}/figures/anti_decreased_ys_yearfe_no_labels.pdf", as(pdf) name("Graph") replace
