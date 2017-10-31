# Recreate the boxplot from Supplemental Figure S4 from Ahrne et al, JPR 2016 (https://www.ncbi.nlm.nih.gov/pubmed/27345528)
library(gdata)

setwd("C:/Users/Judson/Documents/QC metrics/Ahrne 2016")

#dat=read.xls("Ahrne_bhens_1fdr_StdCytActin33176202_metricsPt1.xls")
dat=read.xls("Ahrne_bhens_1fdr_allStds1_metricsPt1.xls")


#order of expected ratios in ratio heatmap columns 1-4 is 0.02/0.1 (5:1); 0.01/0.02 (1:2); 0.015/0.01 (1.5:1); 0.1/0.01 (10:1)
df = data.frame(ratio = c(dat$SILAC.ratio.32.for.user.selected.SILAC.timepoint1,dat$SILAC.ratio.32.for.user.selected.SILAC.timepoint2,
                          dat$SILAC.ratio.32.for.user.selected.SILAC.timepoint3,dat$SILAC.ratio.32.for.user.selected.SILAC.timepoint4),
                Dilution = c(rep('x1vs5',nrow(dat)),rep('x1vs2',nrow(dat)),rep('x1vs1.5',nrow(dat)),rep('x10vs1',nrow(dat)))
                )

df$l2ratio = log(df$ratio,base=2)

#Reorder factors to match the ordering of Ahrne's plot
df$Dilution = factor(df$Dilution, levels = levels(factor(df$Dilution))[c(4,3,2,1)])

median1vs5 = median(df[df$label=='1:5','l2ratio'],na.rm=TRUE)
median1vs2 = median(df[df$label=='1:2','l2ratio'],na.rm=TRUE)
median1vs1.5 = median(df[df$label=='1:1.5','l2ratio'],na.rm=TRUE)
median10vs1 = median(df[df$label=='10:1','l2ratio'],na.rm=TRUE)

###Make plot with ggplot2
library(ggplot2)

p=ggplot(aes(y = l2ratio, x = Dilution, fill = Dilution), data = df) + 
  geom_boxplot(linetype = 'dashed', color = 'black', outlier.shape=1, outlier.size=3) +
  geom_boxplot(aes(ymin=..lower.., ymax=..upper..), outlier.shape='none') +
  scale_fill_manual(values=c('darkgreen', 'blue', 'purple', 'orange')) +
  scale_x_discrete('Dilution', labels=c('1:5','1:2','1:1.5','10:1')) +
	ylab('LFQ ratio (log2)') +
	scale_y_continuous(limits=c(-4,4),breaks=seq(-4,4,by=2),labels=seq(-4,4,by=2)) + 
  geom_hline(yintercept=log(1/5,base=2),linetype='dashed',color='darkgreen', size=0.75) +
  geom_hline(yintercept=log(1/2,base=2),linetype='dashed',color='blue', size=0.75) +
  geom_hline(yintercept=log(1,base=2),linetype='dashed',color='black', size=0.75) +
  geom_hline(yintercept=log(1.5,base=2),linetype='dashed',color='purple', size=0.75) +
  geom_hline(yintercept=log(10,base=2),linetype='dashed',color='orange', size=0.75) +
	theme(
		panel.grid.major = element_blank(),
		panel.grid.minor = element_blank(),
		panel.background = element_blank(),
		panel.border = element_rect(color='black', size=1.25, fill=NA, linetype='solid'),
		axis.text.x = element_text(colour='black', size=22),
		axis.text.y = element_text(colour='black', size=22),
		axis.title.x = element_blank(),
		axis.title.y = element_text(margin=margin(t=0, r=10.5, b=0, l=0), size=26),
		axis.ticks.length = unit(0.33, "cm"),
		legend.position = 'none'
	) +
  labs(x = "Dilution")

ggsave('LFQ accuracy log2 ratio boxplot.png',plot=p)
