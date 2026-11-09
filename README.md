Domain: E-commerce

## How to run

Flipkart E-commerce Decision Intelligence — IIT Mandi Minor in Artificial Intelligence & Data Science capstone. All records are synthetic; these findings do not describe Flipkart's actual customers or performance. Use the `main` branch. Python 3.11.7 and NumPy 2.3.5 were used; Python 3.11 or newer supports the pinned NumPy version.

From a clean clone, create and activate a virtual environment if desired, then run these commands from the repository root in order:

```text
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
python -m unittest discover -s tests -v
```

The unchanged supplied generator prints `800 10 170 c6f249ba693f1ebeed318d6d5e18de23`: rows, columns, churn-positive count and dataset MD5. The committed `data/records.csv` has that exact hash. Git attributes preserve the generator and CSV bytes across platforms. All graded algorithms use our own formulas with Python and NumPy; the permitted linear solve and eigen-decomposition are used in Parts 1 and 3. No external dataset or model service is required. Calculations run offline once NumPy is installed.

`part1.py` writes 47 keys, `part2.py` 61, `part3.py` 48 and `part4.py` 17. Together they generate all 12 required text/CSV files in `outputs/`. Values retain full precision until printing; floats use four decimals. `common.py` shares data integrity checks, training-only standardisation, logistic training, KNN and metrics between Parts 2 and 4. Imports do not execute a Part. Tests regenerate the dataset and all outputs in a fresh temporary directory and compare bytes with the committed artifacts, alongside independent arithmetic and convention checks.

Predictors are X1 tenure, X2 complaint count, X3 discount-order share, X4 historical average order value, X5 loyalty score, X6 cash-on-delivery flag and X7 Plus membership. The outcome is no purchase in the following 90 days. Both value columns are in hundreds of INR; one unit is INR 100. Identifiers are never predictors.

## Part 1 findings

The initial implementation was audited against Tasks 1.1–1.7 before implementing later Parts. All 47 keys, formulas, stable residual ordering and four-decimal formatting were correct, and rerunning regenerated the existing output byte for byte. The README structure was corrected, its unsupported deadline time was removed, and the count of required output files was corrected to 12.

**R1.1.** Reject H0 that the population mean equals INR 800: `z_stat = 2.8930`, exceeding the two-sided critical magnitude 1.96. The sample mean is 8.9382 hundreds of INR, with 95% interval [8.3026, 9.5738]. Its gap above 8.0 is 0.9382 hundreds of INR, approximately INR 93.82. That may affect revenue planning materially, but operational margins are needed to judge practical importance. A Type I error means concluding that the population mean order value differs from INR 800 when that planning assumption is actually correct.

**R1.2.** `n_required = 1293`, compared with 800 available rows: 493 additional observations are needed under the estimated SD to target a 95% margin of INR 50. The current interval is wider than that target.

**R1.3.** `gd_slope_gap = 0.0000` at printing precision. The convex squared-error objective and the prescribed learning rate on standardised X4 let 500 full-batch updates converge to the closed-form line.

**R1.4.** `resid_sd_ratio_raw = 3.7525` shows a funnel with much larger residual spread in the upper fitted-value half. `resid_sd_ratio_log = 1.0396` indicates that the log remedy largely removes that uneven spread under this diagnostic. Raw and logged R² describe different response scales and are not compared.

**R1.5.** Excluding the constant, the coefficients whose absolute value exceeds 0.05 are ln(X4) = 1.1576, X2 = 0.0940, X3 = 0.1067 and X6 = 0.1518. The multiple model's adjusted R² is 0.7403 versus 0.6561 for the log-log baseline on the same logged target, so the extra predictors earn their place as an explanatory baseline. The coefficient cutoff is mechanical: different predictor units prevent interpreting this list as an importance or significance ranking.

## Part 2 findings

Training contains 560 rows and 119 positives; testing contains 240 rows and 51 positives. Each class contributes its first integer-floor 70% to training, and both sets retain file order. All seven features use training means and sample SDs. Logistic regression starts at zero and uses 1,000 simultaneous full-batch updates at learning rate 0.1. KNN folds use training position modulo five, without refitting scaling inside folds. Neighbour ties keep reference file order; K and threshold ties choose the smaller value. Test rows affect none of these choices.

**R2.1.** After 200 updates the training losses, lowest first, are learning rate 1.0: 0.3703; 0.1: 0.3776; 0.01: 0.5091. A small rate progresses slowly, while the larger rate converges faster here; this experiment does not establish that arbitrarily large rates are stable.

**R2.2.** The three largest absolute weights are X3 discount-order share (0.9013), X1 tenure (-0.5554) and X2 complaints (0.5505). The largest odds ratio is X3's 2.4629: a one-training-SD increase in discount-order share multiplies churn odds by 2.4629 with other predictors held fixed; this is an association, not a coupon's causal effect.

**R2.3.** Blind accuracy is 0.7875 versus logistic accuracy 0.8042 at threshold 0.5: many correct negatives can hide missed churners. Lowering the threshold to 0.3 increases recall from 0.4706 to 0.7451 and decreases precision from 0.5455 to 0.4634.

**R2.4.** The selected K is 5. Logistic at 0.3 has F1 0.5714 and recall 0.7451; KNN has F1 0.3696 and recall 0.3333. Logistic is stronger on both measures on this split.

**R2.5.** Recommend logistic regression at the training-selected threshold 0.1500: test cost is INR 6050, compared with INR 11800 at 0.5 and INR 20400 when flagging nobody, using INR 50 per false positive and INR 400 per false negative. Training cost at the selected threshold is INR 12150. Choosing the threshold on training rows preserves the test set as an independent evaluation rather than optimising against its answers. This is a simulation recommendation requiring prospective validation.

## Part 3 findings

**R3.1.** The successive WCSS drops, calculated from `part3_results.txt`'s displayed values, are 659.8282, 108.0739, 171.0575 and 45.5466. The pinned elbow rule supports K = 3 because the 3-to-4 drop is less than half the 2-to-3 drop. The drops do not shrink steadily: pinned starting rows give reproducible solutions, not necessarily global optima. K-means continues until an unchanged assignment pass, retaining empty centroids and choosing the first centroid on distance ties.

**R3.2.** These names and actions come from `part3_segments.csv`:

| Segment | Profile in original units | Business name and proposed action |
|---|---|---|
| A (322) | Tenure 47.5028 months; complaints 1.9503; discount share 0.2250; loyalty 62.1584; churn 0.0186 | Established loyal shoppers: maintain reliable service and loyalty recognition rather than broad retention discounting. |
| B (255) | Average historical value 28.7192 hundreds of INR; complaints 6.0706; churn 0.2353 | High-value shoppers with friction: prioritise complaint resolution. |
| C (223) | Tenure 14.3892 months; discount share 0.7118; loyalty 38.5215; churn 0.4664 | Newer discount-focused shoppers: test targeted onboarding and retention offers. |

C has the highest outcome rate, 0.4664. These are descriptive groups; the data does not establish seasonality or the effectiveness of any proposed action.

**R3.3.** The largest-gap single-linkage cut gives {E1}, {E2, E3, E6}, {E4}, {E5}; the complete-linkage cut gives {E1, E4, E5} and {E2, E3, E6}. Single linkage uses the closest member pair, making the final merge height only 3.4000 and the largest jump occur earlier, between 1.4000 and 2.8000. Complete linkage uses the farthest pair, giving a final height of 7.9000 after 2.9000, so its largest gap separates the two broad groups. Average linkage also gives two clusters at its largest-gap cut.

**R3.4.** The standardised five-feature mean vector is (0, 0, 0, 0, 0), up to floating-point error; PCA explicitly centres it again, builds sample covariance, decomposes it and sorts eigenvalues. Eigenvectors have their largest absolute entry positive. PC1 loads positively on tenure (0.5317) and loyalty (0.4418), and negatively on discount share (-0.5187), complaints (-0.4100) and historical value (-0.2916). Tenure and discount share are its strongest contributors by absolute loading. Three components retain 0.8616 of variance; two retain only 0.7593. Mean PC1 scores A = 1.7773, B = -0.7831 and C = -1.6709 place established loyal shoppers at one end and newer discount-focused shoppers at the other, with B between them. PCA signs do not imply causal effects.

## Part 4 findings

**R4.1.** S4 (tenure ≤ 24 months) wins both information gain (0.1252) and Gini gain (0.0606), separating training churn rates 0.4574 and 0.0887 more strongly than the other candidate questions. A tree grown until every leaf is pure can fit noise and individual customers rather than general patterns; a maximum-depth rule limits that growth.

**R4.2.** Ship logistic regression, with the training-cost threshold 0.1500 described above. In the prescribed comparison at 0.3, logistic has F1 0.5714 and recall 0.7451, compared with KNN 0.3696/0.3333, stump 0.5532/0.7647, hard vote 0.5344/0.6863 and soft vote 0.5507/0.7451 (F1/recall). Combining models did not improve logistic's F1 or recall; the stump recalls slightly more churners but has lower precision. These comparison metrics describe threshold 0.3, not the proposed 0.1500 operating point.

**R4.3.** After round 1, the two misclassified points (3 and 7) each have normalised weight 0.2500, a total of 0.5000. Round 2 trusts h2 more because its weighted error is 0.1667 versus h1's 0.2500, giving alpha 0.8047 versus 0.5493. The final ensemble accuracy is 0.7500. The weights and updates are in `part4_adaboost.csv`.

**R4.4.** Model A option 2 has the following shapes and parameter counts. Spatial layers use height × width × depth; dense layers are vectors. Biases are included; activations and pooling have no trainable parameters.

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1, 5 × 5, stride 1, padding 2 | 28 × 28 × 6 | 156 |
| ReLU | 28 × 28 × 6 | 0 |
| Max-pool, 2 × 2, stride 2 | 14 × 14 × 6 | 0 |
| Conv2, 5 × 5, stride 1, padding 0 | 10 × 10 × 16 | 2416 |
| ReLU | 10 × 10 × 16 | 0 |
| Max-pool, 2 × 2, stride 2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 | 32 | 12832 |
| ReLU | 32 | 0 |
| Dense2 | 10 | 330 |

For a dense layer, parameters = (inputs + 1) × outputs; for convolution, parameters = (kernel height × kernel width × input channels + 1) × filters. The MLP totals 13002 parameters and the CNN 15734 for the specified small architectures. On the larger supplied input, the fully connected layer needs 50332672 parameters versus 1792 for the convolutional layer: local connectivity and shared filters prevent its count from scaling with every input pixel.

Model B uses W_xh shape 128 × 64, W_hh shape 128 × 128, W_hy shape 16 × 128, b_h shape 128 and b_y shape 16, totalling 26768 parameters. Its parameter count does not increase with sequence length because the same recurrent weights are reused at each step, although computation and stored activations do increase. For Model A's ten mutually exclusive classes, use ReLU hidden activations, a softmax output and categorical cross-entropy; Model B's specified recurrent activation is tanh. In a very deep plain network, repeatedly multiplying small derivatives can make early-layer gradients vanish and learning stall. A skip connection adds an identity path that carries gradients more directly to earlier layers.

## Model-choice memo

To: Head of the retention team

Recommend the structured logistic-regression model with threshold 0.1500, selected by minimum training cost. Its test cost is INR 6050 under the supplied INR 50 false-alarm and INR 400 missed-churn rule, compared with INR 11800 at threshold 0.5 and INR 20400 for flagging nobody. These are the Task 2.8 logistic cost comparators; the rule counts false alarms and misses, not a full campaign budget.

In the required Task 4.2 comparison at threshold 0.3, logistic F1 is 0.5714 and recall 0.7451; hard voting yields 0.5344 and 0.6863, and soft voting 0.5507 and 0.7451. The combinations add complexity without improving these measures. The threshold-0.3 metrics are distinct from the proposed threshold-0.1500 operating point.

Treat this as a candidate for a monitored pilot: one synthetic split cannot establish real-world calibration, coupon effectiveness or future costs. Validate prospectively and monitor precision, missed churn and cost drift before broad rollout. Neither Model A for product images nor Model B for review topics is worth building before the structured retention model is in production: they address separate tasks without supplied labelled data or demonstrated retention value.

## Limitations

- The records are synthetic; none of these estimates establishes real Flipkart behaviour or causal effects of coupons.
- The prescribed single file-order split gives only one evaluation; it does not quantify variability across populations or future periods.
- There are only 51 positive test records, so recall and cost are sensitive to a small number of classification changes.
- Cost constants are supplied assumptions. The rule omits true-positive coupon expense, delivery constraints and whether an offer actually prevents churn.
- Regression diagnostics and adjusted fit are in-sample; the value models have no held-out predictive evaluation in this brief.
- K-means uses one pinned initialisation per K and squared distance; segment assignments and elbow behaviour can depend on that choice.
- Standardisation deliberately uses all training rows during KNN cross-validation as prescribed, rather than refitting within each fold. Parts 1 and 3 deliberately use all rows; they do not provide supervised test-set claims.
- Architecture counts assess size only: no image or text model is trained, and parameter counts do not establish predictive quality.
