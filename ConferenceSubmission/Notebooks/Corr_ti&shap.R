# read it to R and coorelation(default)
# scatter plot for each feature in each data
# Rmarkdown file
setwd("/Users/aubrey/Documents/SHK/Dropbox/Dingyi/Data/ti&shap_full")
getwd()

#dataname <- c('abalone', 'bike', 'boston', 'concrete', 'cpu', 'csm', 'fb', 'parkinsons','servo', 'solar','synthetic')

library(ggplot2)

dataname <- c('synthetic1','synthetic2')

Cor <- c()
for(i in dataname){
  ti <- read.csv(file = paste0('ti_',i,'.csv'))[,-c(1)]
  shap <- read.csv(file = paste0('shap_',i,'.csv'))[,-c(1)]
  for(j in colnames(ti)){
    Cor <- c(Cor,as.numeric(cor(ti[j], shap[j])))
    print(paste0("The correlation of treeinterpreter and shap value in ",i,"with regard to ", j, "is ", round(cor(ti[j], shap[j]), digits = 4)))
    temp <- data.frame(c(ti[j],shap[j]))
    colnames(temp) <- c("ti","shap")
    # Basic scatter plot with geom_smooth
    print(ggplot(temp, aes(x=ti, y=shap)) + geom_point() +
            geom_smooth()+
            geom_smooth(method='lm')+
            labs(title=paste0("The scatter plot of treeinterpreter and shap value in ",i,"with regard to ", j),
                 x=paste0("ti_",i,"_",j), y = paste0("shap_",i,"_",j)))
  }
}
  
  