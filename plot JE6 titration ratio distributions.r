# Make histograms displaying the distribution of peak area ratios for experiments
# in which a comparison between the known/expected and the experimentally
# observed fold-changes is desired.

library(readxl)
library(ggplot2)

setwd('C:/Users/jmb/Desktop/LabFiles/Spectral QC')

dat = read_excel('JE6_jmb_AllConc_1FDR_newFWHM_051617_updatedRT raw data dump.xlsx')

makeRatioDistributionPlot = function(ratios, expectedratio, comparison, histcolor, expectedcolor, observedcolor){
  df = data.frame(l2ratio=log(ratios,base=2))
  expectedl2ratio = log(expectedratio,base=2)
  medianl2ratio = median(df$l2ratio, na.rm=TRUE)
  colors=c(histcolor)
  
  expectedlab = paste('Expected:',expectedl2ratio,sep=' ')
  medianlab = paste('Observed:',medianl2ratio,sep=' ')

  p = ggplot(df, aes(x=l2ratio)) +
    geom_histogram(binwidth = 0.2, fill=colors, bins = NULL) +
    scale_x_continuous(breaks=seq(-3,12,3),limits=c(-3,12)) +
    scale_y_continuous(breaks=seq(0,1600,400),limits=c(0,1600)) +
    geom_vline(xintercept=expectedl2ratio,color=expectedcolor,linetype='dashed', size=1.15) +
    geom_vline(xintercept=medianl2ratio,color=observedcolor,linetype='solid', size=1.15) +
    annotate(geom='text', color=c(expectedcolor,observedcolor), size=c(5.5,5.5), x=c(9,9), y=c(1200,1000), label=c(expectedlab,medianlab), color='black',parse=TRUE) +
    labs(x=paste('log2 (',comparison,')',sep='')) + 
    theme_classic() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      axis.line.x = element_line(color='black', size=0.6),
      axis.line.y = element_line(color='black', size=0.6),
      axis.text.x = element_text(colour='black', size=15),
      axis.text.y = element_text(colour='black', size=15),
      axis.title.x = element_text(margin=margin(t=7.5, r=0, b=0, l=0), size=18),  #ditto
      axis.title.y = element_text(margin=margin(t=0, r=5, b=0, l=0), size=18),  #ditto
      axis.ticks.length = unit(0.2, "cm"),
      legend.position = 'none'
    )
  ggsave(filename=paste('JE6 log2 ratio dist ',gsub('/','vs.',comparison),'.png',sep=''), plot=p)
}

makeRatioDistributionPlot(dat$`SILAC ratio 32 for user selected SILAC timepoint1 (1X/0.1X)`, 
  10,'10x/1x peak area','palevioletred','blue1','red')
makeRatioDistributionPlot(dat$`SILAC ratio 32 for user selected SILAC timepoint2 (1X/0.3X)`, 
  (10/3),'10x/3x peak area','mediumslateblue','blue1','red')
makeRatioDistributionPlot(dat$`SILAC ratio 32 for user selected SILAC timepoint3 (0.3X/0.1X)`, 
  3,'3x/1x peak area','yellowgreen','blue1','red')
