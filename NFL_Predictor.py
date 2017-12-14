import HelperFunctions as hf
import pandas as pd

# Create team id lookup
team_dict = hf.create_team_dict()

# File name for raw results
raw_results = "results_data.csv"

# Pull data from pref.com
years = str(list(range(2002, 2017)))
#hf.web_crawler(years, raw_results, team_dict)

raw_data = pd.read_csv(raw_results)
print(raw_data.head())

#data_filename = "results_data_test.csv"
#team_file_stump = directory_stump + "team_data/raw_data_"

#years_array = []
#for year in range(2011, 2012):
#    years_array.append(str(year))

#team_dictionary = create_team_dict()
#create_training_set(years_array, data_filename, team_dictionary)

#results_data = pandas.read_csv(data_filename)
#game_data = pandas.read_csv(directory_stump + "game_dictionary.csv")

'''stats_indices_home = [6, 8, 10, 12, 14]
stats_indices_away = [4, 7, 9, 11, 13]

for i in range(0, 32):
    team_results = pandas.concat([results_data.loc[results_data["Away_Team"] == i], results_data.loc[results_data["Home_Team"] == i]]).sort()
    moving_avgs = moving_average(team_results, 4, stats_indices_home, stats_indices_away)
    filename = team_file_stump + str(i) + ".csv"
    moving_avgs.to_csv(filename, header=True, index_label="Game_Id")'''

#create_game_dictionary(results_data)

'''game_dict = dict()
with open(directory_stump + "/game_dictionary.csv", mode="r") as myfile:
    reader = csv.reader(myfile)
    next(reader, None)
    game_dict = {int(rows[0]):[rows[1], int(rows[2]), rows[3]] for rows in reader}

for i in range(0, 32):
    moving_avgs = pandas.read_csv(team_file_stump + str(i) + ".csv")
    moving_avgs_copy = moving_avgs[:]

    for index, row in moving_avgs.iterrows():
        if game_dict[row[0]][1] < 4:'''

#write_team_stats(results_data)
#write_win_rate(team_file_stump)
#write_elo_values(team_file_stump, results_data)

'''team_elo_dict = list()
for i in range(0, 32):

    elo_dict = dict()
    with open(team_file_stump + "elo_" + str(i) + ".csv", mode="r") as infile:
        reader = csv.reader(infile)
        next(reader, None)
        elo_dict = {row[0]:row[14] for row in reader}

    team_elo_dict.append(elo_dict)

away_elo = list()
home_elo = list()

for index, row in results_data.iterrows():
    away_elo.append(team_elo_dict[row["Away_Team"]][str(index)])
    home_elo.append(team_elo_dict[row["Home_Team"]][str(index)])

results_data["Away_Elo"] = away_elo
results_data["Home_Elo"] = home_elo

#arr = [0,1,2,3,4,5,6,17,18]

#elo_data = results_data.ix[:,arr]
#elo_data.to_csv(directory_stump + "elo_test.csv", header=True, index_label="Game_Id")

elo_data = pandas.read_csv(directory_stump + "elo_test.csv")

scoring = "mean_squared_error"
models = []
models.append(("LR", linear_model.Lasso()))
models.append(("EN", linear_model.ElasticNet()))
models.append(("LinR", linear_model.LinearRegression()))

results = []
names = []

for name, model in models:
    kfold = cv.KFold(n=elo_data.shape[0], n_folds=10, random_state=7)
    cv_results = cv.cross_val_score(model, elo_data.ix[:,8:9], elo_data.ix[:,10].values.ravel(), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(names)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)'''

'''simple_game_results = pandas.read_csv(directory_stump + "game_results_pure.csv")
simple_game_results = simple_game_results[simple_game_results["Year"] > 2013]
train_results = simple_game_results[simple_game_results["Year"] < 2016]
test_results = simple_game_results[simple_game_results["Year"] == 2016]

winner_results_train = pandas.DataFrame()
winner_results_test = pandas.DataFrame()

winner_results_train[["team1", "team2"]] = train_results[["Winning_Team", "Losing_Team"]].copy()
winner_results_train["pred"] = 1
winner_results_test[["team1", "team2"]] = test_results[["Winning_Team", "Losing_Team"]].copy()
winner_results_test["pred"] = 1

loser_results_train = pandas.DataFrame()
loser_results_test = pandas.DataFrame()

loser_results_train[["team1", "team2"]] = train_results[["Losing_Team", "Winning_Team"]].copy()
loser_results_train["pred"] = 0
loser_results_test[["team1", "team2"]] = test_results[["Losing_Team", "Winning_Team"]].copy()
loser_results_test["pred"] = 0

train_data = pandas.concat((winner_results_train, loser_results_train), axis=0)
test_data = pandas.concat((winner_results_test, loser_results_test), axis=0)

n_factors = 75
n = 32

team1_in, t1 = embedding_input("team1_in", n, n_factors, 1e-4)
team2_in, t2 = embedding_input("team2_in", n, n_factors, 1e-4)

b1 = create_bias(team1_in, n)
b2 = create_bias(team2_in, n)

x = merge([t1, t2], mode="dot")
x = Flatten()(x)
x = merge([x, b1], mode="sum")
x = merge([x, b2], mode="sum")
x = Dense(1, activation="sigmoid")(x)
model = Model([team1_in, team2_in], x)
model.compile(Adam(0.001), loss="binary_crossentropy")
model.summary()

train = train_data.values
numpy.random.shuffle(train)
history = model.fit([train[:, 0], train[:, 1]], train[:, 2], batch_size=64, nb_epoch=15, verbose=2)
plt.plot(history.history["loss"])
plt.show()

test_data["model_res"] = model.predict([test_data["team1"], test_data["team2"]])
test_data.to_csv(directory_stump + "CF_results.csv", header=True, index_label="GameId")'''

'''cf_res = pandas.read_csv(directory_stump + "CF_results.csv")

team_elo_dict = list()
for i in range(0, 32):

    elo_dict = dict()
    with open(team_file_stump + "elo_" + str(i) + ".csv", mode="r") as infile:
        reader = csv.reader(infile)
        next(reader, None)
        elo_dict = {row[0]:row[14] for row in reader}

    team_elo_dict.append(elo_dict)

away_elo = list()
home_elo = list()

for index, row in results_data.iterrows():
    away_elo.append(team_elo_dict[row["Away_Team"]][str(index)])
    home_elo.append(team_elo_dict[row["Home_Team"]][str(index)])

results_data["Away_Elo"] = away_elo
results_data["Home_Elo"] = home_elo

arr = [0,1,2,3,4,5,6,17,18]

elo_data = results_data.ix[:,arr]

elo_data = elo_data.ix[3584:,]
writer = csv.writer(open(directory_stump + "cf_test.csv", "w", newline=""))
writer.writerow(["GameId", "Home_Team", "Away_Team", "Home_Elo", "Away_Elo", "CF_res", "Home_Margin"])

for index, row in cf_res.iterrows():
    to_write = list()
    to_write.append(row["GameId"])

    home_team = elo_data.iloc[index, 5]
    away_team = elo_data.iloc[index, 3]

    to_write.append(home_team)
    to_write.append(away_team)

    to_write.append(elo_data.iloc[index, 8])
    to_write.append(elo_data.iloc[index, 7])

    cf = 0
    if row["team1"] == home_team:
        cf = row["model_res"]
    else:
        cf = 1 - row["model_res"]

    to_write.append(cf)
    to_write.append(elo_data.iloc[index, 6] - elo_data.iloc[index, 4])
    writer.writerow(to_write)'''


'''cf_data = pandas.read_csv(directory_stump + "cf_test.csv")

scoring = "neg_mean_squared_error"
models = []
models.append(("LR", linear_model.Lasso()))
models.append(("EN", linear_model.ElasticNet()))
models.append(("LinR", linear_model.LinearRegression()))

results = []
names = []

for name, model in models:
    kfold = cv.KFold(n=cf_data.shape[0], n_folds=10, random_state=7)
    cv_results = cv.cross_val_score(model, cf_data.ix[:,3:6], cf_data.ix[:,6].values.ravel(), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(names)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)'''