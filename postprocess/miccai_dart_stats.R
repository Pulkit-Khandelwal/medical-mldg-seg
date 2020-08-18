## Required libraries
library(tidyverse)
library(ggpubr)
library(rstatix)
library(broom)
library(dplyr)
library(PairedData)

## Read csv with Dice or ASSD scores for each procedure
dataDA <- read.csv("/path/to/metrics_output_for_each_subject.csv")

## Here, there are five procedures: baseline, mldg, k=12, k=-2, and oracle.
## Pairwise Wilcoxon signed-rank test for non-normal data distribution
res <- wilcox.test(dataDA$baseline, dataDA$mldg, paired = TRUE)
res

res <- wilcox.test(dataDA$baseline, dataDA$k1, paired = TRUE)
res

res <- wilcox.test(dataDA$baseline, dataDA$k2, paired = TRUE)
res

res <- wilcox.test(dataDA$baseline, dataDA$oracle, paired = TRUE)
res

res <- t.test(dataDA$baseline, dataDA$mldg, paired = TRUE)
res

## Normality test: plot density
ggdensity(dataDA$mldg,
          main = "Density",
          xlab = "values")

## Normality test: Quantile-Quantile plot
ggqqplot(dataDA$mldg)

## Normality test: Shapiro test. p-value should be >0.05 to say that the data is normally distributed
shapiro.test(dataDA$mldg)
