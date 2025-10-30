PART A: CROSS-CORRELATION ANALYSIS


Cross-Correlation Matrix (rounded to 2 decimal places):
              Fiction  Sci-Fict  Baby_Toddler    Teen   Manga  Art&Hist  ...  Mysteries  Thrillers  Classics  Poetry  Romance  Horror
Fiction          1.00      0.49         -0.05    0.12    0.36     -0.02  ...      -0.42      -0.28     -0.66   -0.04     0.29    0.38
Sci-Fict         0.49      1.00          0.37   -0.38    0.06      0.00  ...      -0.62      -0.45     -0.50    0.01    -0.02    0.25
Baby_Toddler    -0.05      0.37          1.00   -0.62   -0.67     -0.02  ...      -0.49      -0.48     -0.07    0.01    -0.04    0.30
  Teen           0.12     -0.38         -0.62    1.00    0.32      0.02  ...       0.39       0.33     -0.19    0.00     0.55    0.26
 Manga           0.36      0.06         -0.67    0.32    1.00      0.00  ...       0.12       0.19     -0.17   -0.02    -0.09   -0.28
Art&Hist        -0.02      0.00         -0.02    0.02    0.00      1.00  ...       0.01      -0.01      0.02    0.00    -0.01    0.01
SelfImprov      -0.25     -0.62         -0.52    0.69    0.07      0.01  ...       0.58       0.43      0.17   -0.00     0.40    0.10
Cooking         -0.49     -0.33          0.29   -0.13   -0.53      0.01  ...       0.19       0.11      0.40    0.03    -0.01   -0.01
 Games           0.45      0.64          0.61   -0.31   -0.25     -0.02  ...      -0.67      -0.55     -0.59    0.00     0.27    0.59
 Gifts          -0.01     -0.00          0.02   -0.04   -0.00      0.00  ...       0.03      -0.00      0.02   -0.02    -0.04   -0.02
Journals         0.65      0.49         -0.21    0.12    0.51     -0.02  ...      -0.36      -0.19     -0.55   -0.04     0.11    0.15
  News          -0.28      0.06          0.23   -0.56   -0.09      0.00  ...      -0.06      -0.03      0.44    0.03    -0.67   -0.56
NonFict         -0.41     -0.04          0.20   -0.61   -0.10     -0.00  ...       0.03       0.03      0.60    0.04    -0.80   -0.72      
HairyPottery     0.23      0.22         -0.34   -0.16    0.58     -0.01  ...      -0.07       0.06      0.02   -0.01    -0.50   -0.52      
Mysteries       -0.42     -0.62         -0.49    0.39    0.12      0.01  ...       1.00       0.48      0.47    0.02    -0.04   -0.33      
Thrillers       -0.28     -0.45         -0.48    0.33    0.19     -0.01  ...       0.48       1.00      0.34    0.01    -0.06   -0.30      
Classics        -0.66     -0.50         -0.07   -0.19   -0.17      0.02  ...       0.47       0.34      1.00    0.02    -0.52   -0.65      
Poetry          -0.04      0.01          0.01    0.00   -0.02      0.00  ...       0.02       0.01      0.02    1.00    -0.02   -0.02      
Romance          0.29     -0.02         -0.04    0.55   -0.09     -0.01  ...      -0.04      -0.06     -0.52   -0.02     1.00    0.78      
Horror           0.38      0.25          0.30    0.26   -0.28      0.01  ...      -0.33      -0.30     -0.65   -0.02     0.78    1.00      

[20 rows x 20 columns]

Correlation matrix saved to 'correlation_matrix.csv'

PART B: AGGLOMERATIVE CLUSTERING


Starting agglomerative clustering with 1000 records
Building initial distance heap...
Initial heap built. Starting clustering...

Progress: 100/999 merges completed, 901 clusters remaining
Progress: 200/999 merges completed, 801 clusters remaining
Progress: 300/999 merges completed, 701 clusters remaining
Progress: 400/999 merges completed, 601 clusters remaining
Progress: 500/999 merges completed, 501 clusters remaining
Progress: 600/999 merges completed, 401 clusters remaining
Progress: 700/999 merges completed, 301 clusters remaining
Progress: 800/999 merges completed, 201 clusters remaining
Progress: 900/999 merges completed, 101 clusters remaining

Clustering complete! Total merges: 999

LAST 20 MERGES - Smallest Cluster Size Tracking

Merge  980: Cluster    0 (size  330) + Cluster  255 (size    1) -> Smaller cluster size:    1, Distance: 8.01
Merge  981: Cluster    0 (size  331) + Cluster  784 (size    1) -> Smaller cluster size:    1, Distance: 8.10
Merge  982: Cluster    0 (size  332) + Cluster  909 (size    1) -> Smaller cluster size:    1, Distance: 8.13
Merge  983: Cluster  829 (size  187) + Cluster  362 (size    1) -> Smaller cluster size:    1, Distance: 8.14
Merge  984: Cluster    0 (size  333) + Cluster  174 (size    1) -> Smaller cluster size:    1, Distance: 8.19
Merge  985: Cluster  829 (size  188) + Cluster  435 (size    1) -> Smaller cluster size:    1, Distance: 8.23
Merge  986: Cluster    0 (size  334) + Cluster  128 (size    1) -> Smaller cluster size:    1, Distance: 8.36
Merge  987: Cluster  829 (size  189) + Cluster  201 (size    1) -> Smaller cluster size:    1, Distance: 8.39
Merge  988: Cluster    0 (size  335) + Cluster  867 (size    1) -> Smaller cluster size:    1, Distance: 8.44
Merge  989: Cluster   64 (size  218) + Cluster  494 (size    1) -> Smaller cluster size:    1, Distance: 8.44
Merge  990: Cluster    0 (size  336) + Cluster  111 (size    1) -> Smaller cluster size:    1, Distance: 8.44
Merge  991: Cluster  432 (size  248) + Cluster   31 (size    1) -> Smaller cluster size:    1, Distance: 8.52
Merge  992: Cluster   64 (size  219) + Cluster  776 (size    1) -> Smaller cluster size:    1, Distance: 8.56
Merge  993: Cluster    0 (size  337) + Cluster  622 (size    1) -> Smaller cluster size:    1, Distance: 8.79
Merge  994: Cluster    0 (size  338) + Cluster   68 (size    1) -> Smaller cluster size:    1, Distance: 8.79
Merge  995: Cluster  432 (size  249) + Cluster  556 (size    1) -> Smaller cluster size:    1, Distance: 8.94
Merge  996: Cluster    0 (size  339) + Cluster  785 (size    1) -> Smaller cluster size:    1, Distance: 9.42
Merge  997: Cluster  829 (size  190) + Cluster  432 (size  250) -> Smaller cluster size:  190, Distance: 14.10
Merge  998: Cluster  829 (size  440) + Cluster    0 (size  340) -> Smaller cluster size:  340, Distance: 13.59
Merge  999: Cluster  829 (size  780) + Cluster   64 (size  220) -> Smaller cluster size:  220, Distance: 14.44

LAST 10 SMALLEST CLUSTER SIZES IN MERGES:

Sizes: [1, 1, 1, 1, 1, 1, 1, 190, 340, 220]

DENDROGRAM last 20 clusters


Dendrogram saved as 'dendrogram_last_20_clusters.png'

ANALYZING FINAL 4 CLUSTERS


CLUSTER SIZES (from smallest to largest):

Cluster 3: 190 members
Cluster 1: 220 members
Cluster 2: 250 members
Cluster 0: 340 members

CLUSTER PROTOTYPES (Average Attribute Values):


Cluster 0 (Size: 340):
   Attribute  Average Value
     Fiction           2.07
    Sci-Fict           1.97
Baby_Toddler           2.05
        Teen           7.94
       Manga           2.43
    Art&Hist           0.97
  SelfImprov           7.96
     Cooking           5.12
       Games           0.99
       Gifts           4.82
    Journals           1.45
        News           2.18
     NonFict           4.05
HairyPottery           2.56
   Mysteries           6.05
   Thrillers           4.89
    Classics           6.94
      Poetry           4.97
     Romance           5.13
      Horror           2.01

Cluster 1 (Size: 220):
   Attribute  Average Value
     Fiction           2.52
    Sci-Fict           5.85
Baby_Toddler           8.52
        Teen           0.94
       Manga           1.01
    Art&Hist           0.94
  SelfImprov           1.04
     Cooking           4.93
       Games           5.05
       Gifts           4.97
    Journals           2.05
        News           5.27
     NonFict           9.03
HairyPottery           4.48
   Mysteries           2.58
   Thrillers           2.60
    Classics           7.11
      Poetry           5.04
     Romance           0.50
      Horror           0.00

Cluster 2 (Size: 250):
   Attribute  Average Value
     Fiction           4.93
    Sci-Fict           6.05
Baby_Toddler           8.40
        Teen           5.92
       Manga           0.22
    Art&Hist           0.96
  SelfImprov           4.44
     Cooking           4.79
       Games           8.09
       Gifts           4.82
    Journals           2.99
        News           1.04
     NonFict           1.03
HairyPottery           0.52
   Mysteries           1.96
   Thrillers           2.06
    Classics           1.83
      Poetry           4.96
     Romance           8.48
      Horror           9.00

Cluster 3 (Size: 190):
   Attribute  Average Value
     Fiction           7.04
    Sci-Fict           6.87
Baby_Toddler           1.33
        Teen           6.93
       Manga           5.93
    Art&Hist           0.96
  SelfImprov           3.43
     Cooking           1.03
       Games           5.07
       Gifts           4.89
    Journals           6.54
        News           1.95
     NonFict           2.81
HairyPottery           6.57
   Mysteries           2.61
   Thrillers           3.16
    Classics           1.76
      Poetry           4.87
     Romance           4.77
      Horror           3.04

Cluster assignments saved to 'cluster_assignments.csv'


Total records processed: 1000
Number of attributes: 20
Final number of clusters: 4