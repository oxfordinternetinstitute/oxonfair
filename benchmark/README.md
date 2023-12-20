Our initial benchmark follows the four datasets used by Agarwal et al 2018.

They split the datasets into 3/4 train and 1/4 test.
We split them into 1/2 train 1/4 validation 1/4 test.

They report the datsets as (this is directly copied from the paper):
1. The adult income data set (Lichman, 2013) (48,842
examples). Here the task is to predict whether some-
one makes more than $50k per year, with gender as the
protected attribute. To examine the performance for
non-binary protected attributes, we also conducted an-
other experiment with the same data, using both gender
and race (binarized into white and non-white) as the
protected attribute. Relabeling, which requires binary
protected attributes, was therefore not applicable here.
2. ProPublica’s COMPAS recidivism data (7,918 exam-
ples). The task is to predict recidivism from someone’s
criminal history, jail and prison time, demographics,
and COMPAS risk scores, with race as the protected
attribute (restricted to white and black defendants).
3. Law School Admissions Council’s National Longitu-
dinal Bar Passage Study (Wightman, 1998) (20,649
examples). Here the task is to predict someone’s even-
tual passage of the bar exam, with race (restricted to
white and black only) as the protected attribute.
4. The Dutch census data set (Dutch Central Bureau for
Statistics, 2001) (60,420 examples). Here the task is
to predict whether or not someone has a prestigious
occupation, with gender as the protected attribute.

Adult is turned into adult and adult4. 
adult enforces fairness with respect to gender.
adult4 enforces intersectional fairness with respect to (white, non-white) x (male, female).
compas restricts the dataset to white and black only, and enforces with respect ethnicity.
law school race (black, and white only)
census gender

They evaluate logistic regresion and boosting for EO and DP.
All methods have access to the protected attribute.
