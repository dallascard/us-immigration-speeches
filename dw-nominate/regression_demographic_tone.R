library(car)
setwd('/Users/juliame/historical_immigration')

df <- read.table('datasheet_05-02-2022.csv',sep=',',header=TRUE)
df$frac_foreign_pop <- scale(df$frac_foreign_pop)
df$sei <- scale(df$SEI)
df$presgl <- scale(df$PRESGL)
df$year <- scale(df$year)
df$famsize <- scale(df$FAMSIZE)
df$educ <- scale(df$EDUC)
df$lit <- scale(df$LIT)
df$continent <- relevel(factor(df$continent), ref = "Europe")


model <- glm(is_pro ~ 
               year * continent + 
               frac_foreign_pop + 
               famsize + 
               sei
               ,data=df,family='binomial')

model_educ <- glm(is_pro ~ 
               year * continent + 
               frac_foreign_pop + 
               famsize + 
               sei + educ
             ,data=df,family='binomial')

model_lit <- glm(is_pro ~ 
                    year * continent + 
                    frac_foreign_pop + 
                    famsize + 
                    sei + lit
                  ,data=df,family='binomial')

eff <- effect('year*continent',model,xlevels=100)
eff.matrix <- cbind(eff$x,eff$fit,eff$lower,eff$upper)
eff.df <- data.frame(eff.matrix)
#ggplot(eff.df, aes(year,eff.fit, col=continent)) + geom_point()
#+ geom_line() + theme_classic()


#save model coefficients for model, model_educ, and model_lit
#save effdf for main model 
coef.main <- data.frame(summary(model)$coef)
coef.lit<- data.frame(summary(model_lit)$coef)
coef.educ <- data.frame(summary(model_educ)$coef)

write.table(coef.main,file='coef_main.csv',sep=',')
write.table(coef.lit,file='coef_lit.csv',sep=',')
write.table(coef.educ,file='coef_educ.csv',sep=',')
write.table(eff.df,file='year_continent_interaction.csv',sep=',')



