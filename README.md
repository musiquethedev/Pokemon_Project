# Pokemon Viability Analysis and Prediction Project Overview

- Created a machine learning model that predicts whether a Pokemon will be competitively viable
- Scraped over 1000 rows of data from multiple pages on [Serebii.net](https://serebii.net/)
- Pulled and transformed additional data from the established open-source Pokemon Battle simulator Pokemon Showdown (via their respective APIs and GitHub repositories)
- Performed in-depth analysis to understand how features are related and connected with viability
- Tested **Logistic Regression**, **Decision Tree**, **Gaussian Naive Bayes**, **K-Nearest Neighbors**, **Random Forest**, **Support Vector Machine**, and **Gradient-Boosted Tree** algorithms with **Stratified K-Fold Cross Validation** to find which models to explore further
- Optimized **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Gradient-Boosted Tree** models with **Cross-Validated Grid Search** to arrive at the best model
- Analyzed the best model's **Permutation Importance**, **Partial Dependence**, and **Shapley Values** to further understand the connections between the features and viability

## Requirements

Python: >= 3.9

Tcl, TypeScript

Dependencies:

- `beautifulsoup4`
- `joblib`
- `matplotlib`
- `numpy`
- `pandas`
- `requests`
- `scipy`
- `scrapy`
- `seaborn`
- `scikit-learn`
- `tqdm`

## Resources Used

- Scraped [Serebii.net](https://serebii.net/) pages:
	- [National Pokedex](https://www.serebii.net/pokemon/nationalpokedex.shtml)
	- [Paldean Forms](https://www.serebii.net/scarletviolet/paldeanforms.shtml)
	- [Alola Forms](https://www.serebii.net/sunmoon/alolaforms.shtml)
	- [Galarian Forms](https://www.serebii.net/swordshield/galarianforms.shtml)
	- [Mega Evolution](https://www.serebii.net/pokedex-xy/megaevolution.shtml)
- [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown) (GitHub)

## Web Scraping

[`Scraping_and_Wrangling.ipynb`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Scrape_and_Wrangling.ipynb)

Used Scrapy to scrape from the [Serebii.net](https://serebii.net/) pages (see [Resources Used](#Resources-Used)) to obtain the following data on each Pokemon:

- `name`
- `primary type`
- `secondary type`
- `primary ability`
- `secondary ability`
- `hidden ability`
- `hp`
- `attack`
- `defence`
- `special attack`
- `special defense`
- `speed`

For Pokemon that were not scraped from the _National Pokedex_ page, an additional field `alternate` was added and set to `True`.

## Additional Data Collection and Transformation

[`Showdown_API.ipynb`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Showdown_API.ipynb)
[`Add_Tiers.ipynb`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Add_Tiers.ipynb)

The scrape from [Serebii.net](https://serebii.net/) was incomplete, as it had missed the alternate forms of many pokemon.
These alternate forms were added in manually.

Data was also pulled from the Pokemon Showdown Battle Simulator API.
This information had to be heavily transformed:

- Created a column for whether the Pokemon had any special tags (i.e., whether the Pokemon was labelled specifically)
	- Included _Legendary_ and _Mythical_ Pokemon
- Created a column that represented whether the Pokemon was a _Final Evolution_ (transformed from the Pokemon Showdown API's column `evos`)
- Created a row that allowed easy conversion from the Pokemon's official name to the Pokemons's working name in the Pokemon Showdown database

Additional data on the tiering of the Pokemon was pulled directly from a file in the Pokemon Showdown GitHub respotiory (See [Resources Used](#Resources-Used))
The tier of the Pokemon is a ranking of how viable the Pokemon is.
This was transformed from a ranking (`Uber`, `OU`, `UU`, etc.) to an One-versus-All (OVA) approach (_Viable_ versus _Not Viable_).

## EDA

[`EDA.ipynb`](https://github.com/musiquethedev/Pokemon_Project/blob/main/EDA.ipynb)

I looked at the distribution of stats, as well as the relationship between a Pokemon's types, stats, their status as a legendary, and their viability.
Below are some visualization highlights:

![correlation heatmap between each stat](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/Stats_corr.png?raw=True)
![comparison of the distribution of bst for legendary vs nonlegendary pokemon](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/Legend_vs_nonlegend_bst.png?raw=True)
![count of fully evolved pokemon by tier](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/evolved_pkmn_by_tier.png?raw=True)
![comparison of the distribution of bst for pokemon in each tier](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/bst_vs_tier.png?raw=True)
![comparison of the proportion of viable pokemon by type](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/type_viable_table.png?raw=True)

## Model Building

[`Data_Prep.py`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Data_Prep.py)
[`Model_Selection.py`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Model_Selection.py)
[`Model_Tuning.py`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Model_Tuning.py)
[`Model_Creation.py`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Model_Creation.py)

A Preprocessing pipeline was built that dropped the unused features, dropped outliers, minmax-scaled the numerical features (or standard scaled in the case of the Gaussian Naive Bayes model), ordinally encoded boolean features, and one-hot encoded the types. The encoding of the pokemon's type had to be done specially because both type1 and type2 would be encoded in the same columns.

I tried 7 different models and evaluated them using `f1` score.
I used `f1` score because of the unbalanced nature of the data (around 17% of all fully evolved pokemon were viable).
However, I also tracked the model's accuracy, precision, and recall.

Here are the models:

- **Logistic Regression**: Baseline for prediction. Ended up scoring surprisingly well.
- **Decision Tree**: A typical tree-based model. It served as a baseline for the other tree-based models.
- **Gaussian Naive Bayes (GNB)**: A simple mathematical model. Because GNB assumes independent features, I was not confident in its predictive power for this problem.
- **K-Nearest Neighbors (KNN)**: I thought that a good predictor of a pokemon's viability could be other pokemon similar to it. Interestingly, the **KNN** model improved significantly with standard scaling as opposed to minmax scaling.
- **Support Vector Machine**: I tried an SVM, but I was skeptical that it could work because in the EDA, it was evident that a lot of unviable pokemon are similar to viable pokemon stat-wise.
- **Gradient-Boosted Tree** (from `sklearn`): Because of the success of the **Decision Tree** and the **Random Forest**, I added in a simple **Gradient-Boosted Tree**, since they're typically powerful right out of the box.

Before tuning, here are the `f1` scores of the models.
(`f1` is a measure of the harmonic mean of **Precision** and **Recall** and ranges from `0` to `1`. `f1` scores change slightly with each evaluation):

| Logistic Regression | Decision Tree | Naive Bayes | Random Forest | K-Nearest Neighbor | Support Vector Machine | Gradient-Boosted Tree |
| :-----------------: | :-----------: | :---------: | :-----------: | :----------------: | :--------------------: | :-------------------: |
| 0.643               | 0.632         | 0.571       | 0.690         | 0.472              | 0.575                  | 0.670                 |

It's interesting to note that while nearly all the models had a much higher precision than recall, the **Naive Bayes** model had the reverse: a very good recall score of `0.702`.

After running several runs of the **Stratified K-Fold Cross Validation** and finding the mean `f1` score for each model for each run, I decided to continue tuning with the **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Gradient-Boosted Tree** models, using **Cross-Validation Grid Search**.

Scores post-tuning on validation data:

| Logistic Regression | Decision Tree | Random Forest | Gradient-Boosted Tree |
| :-----------------: | :-----------: | :-----------: | :-------------------: |
| 0.769               | 0.700         | 0.800         | 0.737                 |

Ultimately, the **Random Forest** model outperformed the other models, not only during testing and validation, but also in that it was less likely to be horribly wrong (less variation in `f1` scores).

## Model Analysis

[`Model_Analysis.ipynb`](https://github.com/musiquethedev/Pokemon_Project/blob/main/Model_Analysis.ipynb)

After creating the Random Forest model and training it on the entire dataset, I used several different techniques to evaluate how the model made its decision. Here are some highlights from that process.

![partial dependence plot for bst](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/bst_partial_dependence.png?raw=True)
![shapley value force plot for a specific example](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/shap_force.png?raw=True)
![shapley value summary plot](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/shap_summary.png?raw=True)
![shapley value dependence plot for bst and speed](https://github.com/musiquethedev/Pokemon_Project/blob/main/images/shap_dependence.png?raw=True)

## Results

Ultimately, I succeeded in creating a model that could semi-reliably predict the viability of a Pokemon (the `f1` scores of each model were prone to changing across multiple successive runs of StratifiedKFold Cross Validation). However, in collecting data and processing it for the model, I had deliberately omitted several factors:

- **Ability**: a Pokemon's best ability has a huge impact on its viability. A bad ability can render a great Pokemon useless, while an amazing ability can render a terrible Pokemon incredibly powerful
- **Movepool**: some Pokemon are not great stat-wise, but have access to amazing moves that allow them to provide great utility or have enormous power.
- **Context**: The viability of a Pokemon is judged within the context of the metagame: the other Pokemon it has to play with and against. 

These factors were omitted for simplicity in data collection and for time reasons. Their omission means that any model, no matter how tuned, will never be able to perfectly predict the viability of pokemon with the given dataset.
However, the models performed noticeably higher than baseline, meaning that the given data--stats, legendary status, and typing--still plays a significant role in the viability of Pokemon.
This goes against common consensus in the Pokemon competitive community, where movepool and abilities are often favored over stats.

## Next Steps

To continue this project further, additional data--such as the missing data mentioned above--could be collected and a new model created that inputs these features. There could also be additional experimentation with different ways to encode the "type" feature.
Finally, that model could be productionalized into a tool that helps Pokemon players easily determine the strength of different Pokemon at the start of a generation, when such information is not yet common knowledge.

## Additional Attributions

- My father wrote the Tcl script to convert the TypeScript file from pokemon showdown into JSON so that it could be easily imported into Python
- ChatGPT for various odds and ends as well as general understanding
- [PlayingNumbers/ds_salary_proj](https://github.com/PlayingNumbers/ds_salary_proj/blob/master/README.md) (GitHub) by Ken Jee as a README template
