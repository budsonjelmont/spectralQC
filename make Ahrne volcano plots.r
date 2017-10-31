# Recreate the volcano-type plot from Figure 2C from Ahrne et al, JPR 2016 (https://www.ncbi.nlm.nih.gov/pubmed/27345528)

library(gdata)

setwd('C:/Users/Judson/Documents/QC metrics/Ahrne 2016')

dat_hsap = read.xls('Ahrne_hsapiens_1fdr_allStds1_metricsPt1.xls')
dat_bhen = read.xls('Ahrne_bhens_1fdr_allStds1_metricsPt1.xls')

dat_hsap$species = 'Human'
dat_bhen$species = 'B.henselae'

dat = rbind(dat_hsap, dat_bhen)

doAhrnePlot = function(df, title, timepoint, expectedratio){
  
  df$ratios = df[eval(paste('SILAC.ratio.32.for.user.selected.SILAC.timepoint',timepoint,sep=''))]
  df = df[!is.na(df$ratios),]
  df$l2ratios = log(df$ratios,base=2)[,1]

  l2medianratio = median(df$l2ratios[df$species=='B.henselae'])  
  
  df$qvals = df[eval(paste('qvalues.for.SILAC.timepoint',timepoint,sep=''))]
  df$l10qvals = -1*log(df$qvals,base=10)[,1]

  df$color[which(df$species=='B.henselae')] = 'firebrick2'
  df$color[which(df$species=='Human')] = 'black'

  gp=ggplot(df, aes(x=l2ratios, y=l10qvals, color=color)) +
    scale_color_identity() +
    geom_point() +
    xlab('LFQ Ratio (log2)') +
    ylab('qValue (-log10)') +
    geom_vline(xintercept=log(expectedratio,base=2),linetype='dashed',color='red', size=1.5) +
    geom_vline(xintercept=l2medianratio,linetype='solid',color='red', size=1.25) +
    scale_x_continuous(limits=c(-0.5,1),labels=format(seq(-0.5,1,by=0.5), nsmall=1),breaks=seq(-0.5,1,by=0.5)) +
    scale_y_continuous(limits=c(0,10),labels=seq(0,10,by=2),breaks=seq(0,10,by=2)) +
    theme_classic() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      panel.border = element_rect(color='black', size=1.25, fill=NA, linetype='solid'),
      axis.text.x = element_text(colour='black', margin=margin(t=0, r=0, b=0, l=0), size=32),
      axis.text.y = element_text(angle = 90, colour='black', margin=margin(t=0, r=2, b=0, l=0), size=32),
      axis.title.x = element_text(margin=margin(t=10, r=0, b=0, l=0), size=34),
      axis.title.y = element_text(margin=margin(t=0, r=11.5, b=0, l=0), size=34),
      axis.ticks.length = unit(0.33, "cm"),
      legend.position = 'left'
    )
  
  ggsave(paste(title,'.png',sep=''),plot=gp)
}

#order of expected ratios in ratio heatmap columns 1-4 is 0.1/0.02 (5:1); 0.01/0.02 (1:2); 0.015/0.01 (1.5:1); 0.1/0.01 (10:1)
doAhrnePlot(dat, 'Ahrne LFQ ratio plot--1.5 ratio', 3, 1.5)

