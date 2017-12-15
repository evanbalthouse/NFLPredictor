import HelperFunctions as hf
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup, Comment
import requests, re
import pandas
import csv
import numpy
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn import linear_model
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model

# Creates team id lookup table
def create_team_dict():
    team_dict = {}
    team_dict["Carolina Panthers"] = 0
    team_dict["Denver Broncos"] = 1
    team_dict["Green Bay Packers"] = 2
    team_dict["Jacksonville Jaguars"] = 3
    team_dict["Buffalo Bills"] = 4
    team_dict["Baltimore Ravens"] = 5
    team_dict["Tampa Bay Buccaneers"] = 6
    team_dict["Atlanta Falcons"] = 7
    team_dict["Oakland Raiders"] = 8
    team_dict["New Orleans Saints"] = 9
    team_dict["Chicago Bears"] = 10
    team_dict["Houston Texans"] = 11
    team_dict["Cleveland Browns"] = 12
    team_dict["Philadelphia Eagles"] = 13
    team_dict["Cincinnati Bengals"] = 14
    team_dict["New York Jets"] = 15
    team_dict["San Diego Chargers"] = 16
    team_dict["Kansas City Chiefs"] = 17
    team_dict["Minnesota Vikings"] = 18
    team_dict["Tennessee Titans"] = 19
    team_dict["Miami Dolphins"] = 20
    team_dict["Seattle Seahawks"] = 21
    team_dict["Detroit Lions"] = 22
    team_dict["Indianapolis Colts"] = 23
    team_dict["New York Giants"] = 24
    team_dict["Dallas Cowboys"] = 25
    team_dict["New England Patriots"] = 26
    team_dict["Arizona Cardinals"] = 27
    team_dict["Pittsburgh Steelers"] = 28
    team_dict["Washington Redskins"] = 29
    team_dict["Los Angeles Rams"] = 30
    team_dict["San Francisco 49ers"] = 31
    team_dict["St. Louis Rams"] = 30
    return team_dict;

# Crawls pref.com for the given years and prints results to a .csv with the given name
def web_crawler(years, filename, team_dict):
    weeks = []
    for i in range(3, 4):
        weeks.append("week_" + str(i))

    to_write = list()
    to_write.append(["Year", "Week", "Date", "Away_Team", "Away_Score", "Home_Team", "Home_Score", "Away_First_Downs",
                     "Home_First_Downs", "Away_Rushing_Yards", "Home_Rushing_Yards", "Away_Passing_Yards",
                     "Home_Passing_Yards", "Away_Turnovers", "Home_Turnovers", "Favored_Team", "Vegas_Line"])

    base_url = "http://static.pfref.com"
    base_url_to_scrape = base_url + "/years/"

    for year in years:
        year_url_to_scrape = base_url_to_scrape + year + "/"

        for week in weeks:
            url_to_scrape = year_url_to_scrape + week + ".htm"
            print("Loading week " + str(week.split("_")[1]) + " of the " + str(year) + " season.")
            r = requests.get(url_to_scrape)

            soup = BeautifulSoup(r.text, "html.parser")

            for all_games in soup.select(".game_summaries"):

                for game in all_games.select(".teams"):
                    game_data_to_write = list()
                    game_data_to_write.append(year)
                    game_data_to_write.append(week)
                    game_data = game.findAll("tr")

                    box_url = ""

                    for data in game_data:
                        date = data.string

                        if date is not None:
                            game_data_to_write.append(str(date))
                        else:
                            team = team_dict[data.find("td").string]
                            game_data_to_write.append(str(team))
                            game_data_to_write.append(str(data.find("td", {"class": "right"}).string))

                            box_score = data.find("td", {"class": "right gamelink"})
                            if box_score is not None:
                                box_url = base_url + box_score.a.get("href")

                    get_team_stats_data(box_url, game_data_to_write, team_dict)
                    to_write.append(game_data_to_write)

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(to_write)

# Helper function for web_crawler
def get_team_stats_data(url_to_load, data_to_write, team_dict):
    r = requests.get(url_to_load)
    soup = BeautifulSoup(r.text, "html.parser")

    comments = soup.findAll(text=lambda text: isinstance(text, Comment))
    rx = re.compile(r'<table.+?id="team_stats".+?>[\s\S]+?</table>')

    table = ""
    for comment in comments:
        try:
            table = rx.search(comment.string).group(0)
            break
        except:
            pass
    bs_table = BeautifulSoup(table, "html.parser")
    stats_indices = [0, 1, 2, 3, 8, 9, 14, 15]
    table_stats = bs_table.find_all("td", {"class": "center"})

    for i in stats_indices:
        if i == 2 or i == 3:
            data_to_write.append(table_stats[i].string.split("-")[1])
        else:
            data_to_write.append(table_stats[i].string)

    comments = soup.findAll(text=lambda text: isinstance(text, Comment))
    rx = re.compile(r'<table.+?id="game_info".+?>[\s\S]+?</table>')
    table = ""
    for comment in comments:
        try:
            table = rx.search(comment.string).group(0)
            break
        except:
            pass

    vl_table = BeautifulSoup(table, "html.parser")
    table_stats = vl_table.find_all("td", {"class": "center"})
    line = table_stats[len(table_stats) - 2].string
    line_array = line.split(" ")
    team = " ".join(line_array[0:len(line_array) - 1])
    points = str(line_array[-1])

    if line != "Pick":
        data_to_write.append(team_dict[team])
        data_to_write.append(points)
    else:
        data_to_write.append(-1)
        data_to_write.append(0)
    return data_to_write[:]

# Slice dataframe with the given years, inclusive on begin, exclusive on end
def year_slice(raw_dataframe, begin_year, end_year):
    return raw_dataframe[(raw_dataframe["Year"] >= begin_year) & (raw_dataframe["Year"] < end_year)]

# Initialize dataset of features for modeling
def create_feature_df():
    columns = ["Week_ID", "Away_Team", "Away_Score", "Home_Team", "Home_Score",
               "Away_PPG_ma", "Home_PPG_ma", "Away_PPGA_ma", "Home_PPGA_ma",
               "Away_FD_ma", "Home_FD_ma", "Away_FDA_ma", "Home_FDA_ma",
               "Away_RYPG_ma", "Home_RYPG_ma", "Away_RYPGA_ma", "Home_RYPGA_ma",
               "Away_PYPG_ma", "Home_PYPG_ma", "Away_PYPGA_ma", "Home_PYPGA_ma",
               "Away_TO_ma", "Home_TO_ma", "Away_TOA_ma", "Home_TOA_ma",
               "Away_ELO", "Home_ELO", "Away_WR", "Home_WR", "CF_Model_Results", "Vegas_Line"]
    return pandas.DataFrame(columns=columns)

# Return team specific data in an analysis friendly manner - with for-against format vs home-away format
def get_team_results(raw_data, team_id):
    team_specific_results = raw_data.loc[(raw_data["Home_Team"] == team_id) | (raw_data["Away_Team"] == team_id)]

    cols = ["Week_ID", "Year", "Week", "Date", "For_Team", "For_Score",
     "For_First_Downs", "For_Rushing_Yards", "For_Passing_Yards", "For_Turnovers",
     "Against_Team", "Against_Score", "Against_First_Downs", "Against_Rushing_Yards",
     "Against_Passing_Yards", "Against_Turnovers"]

    to_return = list()

    for index, row in team_specific_results.iterrows():
        if row["Away_Team"] == team_id:
            to_return.append(row[["Week_ID", "Year", "Week", "Date", "Away_Team", "Away_Score", "Away_First_Downs", "Away_Rushing_Yards",
                                                                                "Away_Passing_Yards", "Away_Turnovers", "Home_Team", "Home_Score", "Home_First_Downs", "Home_Rushing_Yards",
                                                                                "Home_Passing_Yards", "Home_Turnovers"]].tolist())
        elif row["Home_Team"] == team_id:
            to_return.append(row[["Week_ID", "Year", "Week", "Date", "Home_Team", "Home_Score", "Home_First_Downs", "Home_Rushing_Yards",
                                                                                "Home_Passing_Yards", "Home_Turnovers", "Away_Team", "Away_Score", "Away_First_Downs", "Away_Rushing_Yards",
                                                                                "Away_Passing_Yards", "Away_Turnovers"]].tolist())

    return pandas.DataFrame(to_return, columns=cols)









# Create team id lookup
team_dict = create_team_dict()

# File name for raw results
raw_results = "results_data_full.csv"

# Rolling average length
rolling_avg_window = 4

# Initialize years list to pull data for
years = list(map(str, list(range(2002, 2017))))

# Either crawl web to populate data, or read in .csv containing raw data
#hf.web_crawler(years, raw_results, team_dict)
raw_data = pd.read_csv(raw_results)

# Create week_id column
raw_data["Week_ID"] = range(raw_data.shape[0])
#print(raw_data.head())

# Create DataFrame to hold future features
feature_df = create_feature_df()

# Initialize feature DataFrame columns
feature_df[["Week_ID", "Away_Team", "Away_Score", "Home_Team",
            "Home_Score", "Favored_Team", "Vegas_Line"]] = raw_data[["Week_ID", "Away_Team", "Away_Score", "Home_Team",
                                                                     "Home_Score", "Favored_Team", "Vegas_Line"]]

# Double check on get_team_results function, should be 16 games * len(years)
for i in range(32):
    assert get_team_results(raw_data, i).shape == (16 * len(years), 16)

# Double check on year_slice function, should be 16 games * 32 teams / 2 for duplicates per year
for i in years:
    assert year_slice(raw_data, int(i), int(i) + 1).shape == (256, 18)

# Create dictionary holding moving average stats for each team
team_results_dict = dict()
for i in range(32):
    temp_results = get_team_results(raw_data, i)
    temp_results["For_PPG"] = temp_results["For_Score"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["For_FD"] = temp_results["For_First_Downs"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["For_RYPG"] = temp_results["For_Rushing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["For_PYPG"] = temp_results["For_Passing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["For_TO"] = temp_results["For_Turnovers"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()

    temp_results["Against_PPG"] = temp_results["Against_Score"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["Against_FD"] = temp_results["Against_First_Downs"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["Against_RYPG"] = temp_results["Against_Rushing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["Against_PYPG"] = temp_results["Against_Passing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    temp_results["Against_TO"] = temp_results["Against_Turnovers"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean()
    team_results_dict[i] = temp_results[["Week_ID", "Year", "Week", "Date", "For_Team",
                                         "For_PPG", "For_FD", "For_RYPG", "For_PYPG", "For_TO",
                                         "Against_Team", "Against_PPG", "Against_FD", "Against_RYPG", "Against_PYPG", "Against_TO"]]

# Check on results for team_results_dict
for i in range(32):
    assert team_results_dict[i].shape == (16 * len(years), 16)
    assert team_results_dict[i].isnull().sum().sum() == ((rolling_avg_window - 1) * 10)

























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