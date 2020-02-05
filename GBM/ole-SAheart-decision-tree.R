###**********************************************************************###
### R SCRIPT:  ole-SAheart-decision-tree.R                               ###
###                                                                      ###
### AUTHOR  :  OLE                                                       ###
###                                                                      ###
### DATE    :  07OCT2019                                                 ###
###                                                                      ###
### PROJECT :  ML project                                                ###
###                                                                      ###
### DATA INTRO:>> A retrospective sample of males in a heart-disease     ###
###            high-risk region of the Western Cape, South Africa. There ###
###            are roughly two controls per case of CHD. Many of the     ###
###            CHD positive men have undergone blood pressure reduction  ###
###            treatment and other programs to reduce their risk factors ###
###            after their CHD event. In some cases the measurements were###
###            made after these treatments. These data are taken from a  ###
###            larger dataset, described in  Rousseauw et al, 1983,      ###
###            South African Medical Journal <<                          ### 
###                                                                      ###
### VARIABLES: sbp		systolic blood pressure                  ###
###            tobacco		cumulative tobacco (kg)                  ###
###            ldl		low density lipoprotein cholesterol      ###
###            adiposity                                                 ###
###            famhist		family history of heart disease          ###
###                                           (Present, Absent)          ###
###            typea		type-A behavior                          ###
###            obesity                                                   ###
###            alcohol		current alcohol consumption              ###
###            age		age at onset                             ###
###            chd		response, coronary heart disease         ###
###                                                                      ###
### PURPOSE :  Reproduce summary stats from ElemStatLearn Book, and use  ###
###            data as example in ML course                              ###
###                                                                      ###
### REF.    :  Book:  Hastie, Tibshirani, and Friedman (2nd ed.)         ###
###                                                                      ###
### PROGRAM HISTORY                                                      ###
### VERSION :  07OCT2019 - set up                                        ###
###                                                                      ###
###**********************************************************************###


#+--------------------------------------------------------+
#| If run on local PC:                                    |
#| Change drive according to drive letter applied on PC   |
#| The rest of the input/output path is set automatically |
#| by function file.path below                            |
#+--------------------------------------------------------+
drive <- "C:"


#+-----------------------------------+  
#|                                   |
#|            Functions              |
#|                                   |
#+-----------------------------------+  
path2file <- function(drive,p1,p2,file)  file.path(drive,p1,p2,file)
path3file <- function(drive,p1,p2,p3,file)  file.path(drive,p1,p2,p3,file)


#+-----------------------------------+  
#|                                   |
#|              Rout                 |
#|                                   |
#+-----------------------------------+  
# Note: either run program interactively with 'sink' command
#       or non-interactively, without 'sink' (since the .Rout
#       file will be generated automatically if batch job)
outfile <- path2file(drive,"MLcourse","ElemStatLearn","ole-SAheart-decision-tree.Rout")
sink(file=outfile)


#+-----------------------------------+  
#|                                   |
#|             OPTIONS               |
#|                                   |
#+-----------------------------------+                                                                           
options(verbose=TRUE, echo=TRUE, digits=3)
op <- options(); utils::str(op)



#+-----------------------------------+  
#|                                   |
#|           LOAD PACKAGES           |
#|                                   |
#+-----------------------------------+
#require(graphics)
#require(grDevices)
#require(lattice)


#+-----------------------------------+  
#|                                   |
#|            READ DATA              |
#|                                   |
#+-----------------------------------+
infile1  <- path3file(drive,"MLcourse","ElemStatLearn","datasets","SAheart_data.txt")
heartdat <- read.table(infile1,sep=",",header=TRUE,row.names=1)


#+-----------------------------------+  
#|                                   |
#|    LIST FIRST ROWS OF DATASET     |
#|                                   |
#+-----------------------------------+
head(heartdat)



# --- terminate log file ---
sink()

#+-----------------------------------+  
#|                                   |
#|          END OF SCRIPT            |
#|                                   |
#+-----------------------------------+  