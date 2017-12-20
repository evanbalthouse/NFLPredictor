import pandas as pd
import requests
import re
from sklearn import linear_model
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from bs4 import BeautifulSoup, Comment
import requests, re
import pandas
import csv
import numpy
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
pd.options.mode.chained_assignment = None


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
    for i in range(1, 18):
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
    # Re-order columns - easier to do it this way then re-tool the web scraping code
    data = pd.read_csv(filename)
    data = data[["Year", "Week", "Date", "Away_Team", "Away_Score", "Away_First_Downs", "Away_Rushing_Yards", "Away_Passing_Yards", "Away_Turnovers",
                 "Home_Team", "Home_Score", "Home_First_Downs", "Home_Rushing_Yards", "Home_Passing_Yards", "Home_Turnovers",
                 "Favored_Team", "Vegas_Line"]]
    data.to_csv(filename)


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


# Return team specific data in an analysis friendly manner - with for-against format vs home-away format
def get_team_results(raw_data, team_id):
    team_specific_results = raw_data.loc[(raw_data["Home_Team"] == team_id) | (raw_data["Away_Team"] == team_id)]

    cols = ["Week_ID", "Year", "Week", "Date", "For_Team", "For_Score",
     "For_First_Downs", "For_Rushing_Yards", "For_Passing_Yards", "For_Turnovers",
     "Against_Team", "Against_Score", "Against_First_Downs", "Against_Rushing_Yards",
     "Against_Passing_Yards", "Against_Turnovers", "Result"]

    to_return = list()

    for index, row in team_specific_results.iterrows():
        if row["Away_Team"] == team_id:
            row_to_append = row[["Week_ID", "Year", "Week", "Date", "Away_Team", "Away_Score", "Away_First_Downs", "Away_Rushing_Yards",
                                                                                "Away_Passing_Yards", "Away_Turnovers", "Home_Team", "Home_Score", "Home_First_Downs", "Home_Rushing_Yards",
                                                                                "Home_Passing_Yards", "Home_Turnovers"]].tolist()

        elif row["Home_Team"] == team_id:
            row_to_append = row[["Week_ID", "Year", "Week", "Date", "Home_Team", "Home_Score", "Home_First_Downs", "Home_Rushing_Yards",
                                                                                "Home_Passing_Yards", "Home_Turnovers", "Away_Team", "Away_Score", "Away_First_Downs", "Away_Rushing_Yards",
                                                                                "Away_Passing_Yards", "Away_Turnovers"]].tolist()
        if row_to_append[5] > row_to_append[11]:
            row_to_append.append(1)
        elif row_to_append[5] < row_to_append[11]:
            row_to_append.append(0)
        else:
            row_to_append.append(0.5)
        to_return.append(row_to_append)
    return pandas.DataFrame(to_return, columns=cols)


# Return list of lists with elo ratings per team per game, uses 538's elo rating for NFL games
def calculate_elo_values(features_df):
    team_elo_list = list()

    for i in range(len(team_dict)):
        team_elo = list()
        team_elo.append(1500)
        team_elo_list.append(team_elo)

    for index, row in features_df.iterrows():
        away_team = row["Away_Team"]
        home_team = row["Home_Team"]
        away_score = row["Away_Score"]
        home_score = row["Home_Score"]

        res = 1
        if away_score < home_score:
            res = 0
        elif away_score == home_score:
            res = 0.5

        away_elo_series = team_elo_list[away_team]
        home_elo_series = team_elo_list[home_team]

        away_elo_prev = away_elo_series[len(away_elo_series) - 1]
        home_elo_prev = home_elo_series[len(home_elo_series) - 1]

        if row["Week"] == 1 and row["Year"] != 2002:
            y_away = 0.5 * (1500 - away_elo_prev)
            y_home = 0.5 * (1500 - home_elo_prev)

            away_elo_prev += y_away
            home_elo_prev += y_home

        r_winning = away_elo_prev
        r_losing = home_elo_prev
        # If tie, result does not matter, margin of victory multiplier is 0
        if res == 0:
            r_winning = home_elo_prev
            r_losing = away_elo_prev

        mov_mult = numpy.log(abs(away_score - home_score) + 1) * (2.2 / (0.001 * (r_winning - r_losing) + 2.2))

        express_away = 1 / (1 + 10 ** ((away_elo_prev - home_elo_prev) / 400))
        express_home = 1 / (1 + 10 ** ((home_elo_prev - away_elo_prev) / 400))

        away_elo_new = away_elo_prev + 20 * mov_mult * (res - express_away)
        home_elo_new = home_elo_prev + 20 * mov_mult * ((1 - res) - express_home)

        team_elo_list[away_team].append(away_elo_new)
        team_elo_list[home_team].append(home_elo_new)
    return team_elo_list


# Maps Elo rating grid to away/home team in data format
def get_away_home_elos(feature_df, elo_dict):
    away = list()
    home = list()

    elo_indices = [0] * len(team_dict)

    for index, row in feature_df.iterrows():
        away_team = row["Away_Team"]
        home_team = row["Home_Team"]

        away_elo = elo_dict[away_team]
        home_elo = elo_dict[home_team]

        away.append(away_elo[elo_indices[away_team]])
        home.append(home_elo[elo_indices[home_team]])

        elo_indices[away_team] += 1
        elo_indices[home_team] += 1

    return away, home


# Creates simple dataframe of [team_1, team_2, result={0, 0.5, 1}]
def get_simple_results(raw_data, begin_year, end_year):
    simple_df = pandas.DataFrame(columns=["For_Team", "Against_Team", "Result"])

    for i in range(len(team_dict)):
        team_results = get_team_results(raw_data, i)
        year_data = year_slice(team_results, begin_year, end_year)

        simple_df = simple_df.append(year_data[["For_Team", "Against_Team", "Result"]])

    return simple_df.rename(columns={"For_Team": "team1", "Against_Team": "team2", "Result": "pred"})


# CF helper functions
def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype="int64", name=name)
    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)


def create_bias(inp, n_in):
    x = Embedding(n_in, 1, input_length=1)(inp)
    return Flatten()(x)


# Create CF Model from pure game results [team1, team2, result={0, 0.5, 1}]
def get_cf_model(pure_results):
    pure_results.index = range(len(pure_results.index))
    train = pure_results.values
    numpy.random.shuffle(train)

    n = 32
    n_factors = 16

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

    model.fit([train[:, 0], train[:, 1]], train[:, 2], batch_size=64, nb_epoch=70, verbose=0)
    return model


# Print out results in test set
def print_results(test_set, pred_set, bet, reg_flag):
    num_correct = 0
    running_gains = 0
    bet = 11

    test_set.index = range(len(test_set.index))

    if reg_flag:

        for index, row in test_set.iterrows():
            v_line = row["Vegas_Line"]

            pred_value = pred_set[index]
            truth = row["Point_Differential"]
            if ((pred_value > v_line) and (truth > v_line)) or (((pred_value < v_line) and (truth < v_line))):
                running_gains += bet * 0.909091
                num_correct += 1
            else:
                running_gains -= bet

        print("Num correct: " + str(num_correct) + " out of " + str(len(pred_set)) + ", " + str(
            num_correct / len(pred_set)) + "%")
        print("On a bet of $" + str(bet) + " per game, winnings of $" + str(running_gains))
    else:
        for index, row in test_set.iterrows():
            v_line = row["Vegas_Line"]

            pred_value = pred_set[index]
            truth = row["Point_Differential"]
            if ((pred_value == 1) and (truth > v_line)) or (((pred_value == 0) and (truth < v_line))):
                running_gains += bet * 0.909091
                num_correct += 1
            else:
                running_gains -= bet

        print("Num correct: " + str(num_correct) + " out of " + str(len(pred_set)) + ", " + str(
            num_correct / len(pred_set)) + "%")
        print("On a bet of $" + str(bet) + " per game, winnings of $" + str(running_gains))

# Calculate winning percentage and bet winnings for each year in given range
def performance_by_year(begin_year, end_year, feature_dataset, full_dataset, feature_columns, reg_flag):
    winnings = list()
    corrects = list()
    for i in range(begin_year, end_year):
        testing_begin = i
        testing_end = i + 3

        training_features = year_slice(feature_dataset, testing_begin, testing_end + 1)
        testing_features = year_slice(feature_dataset, testing_end + 1, testing_end + 2)

        full_results = get_simple_results(full_dataset, testing_begin, testing_end + 1)

        model = get_cf_model(full_results)

        training_features.loc[:, "CF_Result"] = model.predict(
            [training_features["Away_Team"], training_features["Home_Team"]])
        training_features.loc[:, "Point_Differential"] = (
        training_features["Away_Score"].copy() - training_features["Home_Score"].copy())

        testing_features.loc[:, "CF_Result"] = model.predict(
            [testing_features["Away_Team"], testing_features["Home_Team"]])
        testing_features.loc[:, "Point_Differential"] = (
        testing_features["Away_Score"].copy() - testing_features["Home_Score"].copy())

        training_features["Vegas_Line"][training_features["Favored_Team"] == training_features["Away_Team"]] = -1 * \
                                                                                                               training_features[
                                                                                                                   "Vegas_Line"][
                                                                                                                   training_features[
                                                                                                                       "Favored_Team"] ==
                                                                                                                   training_features[
                                                                                                                       "Away_Team"]]
        testing_features["Vegas_Line"][testing_features["Favored_Team"] == testing_features["Away_Team"]] = -1 * \
                                                                                                            testing_features[
                                                                                                                "Vegas_Line"][
                                                                                                                testing_features[
                                                                                                                    "Favored_Team"] ==
                                                                                                                testing_features[
                                                                                                                    "Away_Team"]]

        if reg_flag:

            from sklearn import preprocessing

            min_max_scaler = preprocessing.MinMaxScaler()
            transformed_training = min_max_scaler.fit_transform(training_features[feature_columns[8:33]])
            transformed_testing = min_max_scaler.transform(testing_features[feature_columns[8:33]])

            model = MLPRegressor()
            lm = model.fit(transformed_training, training_features["Point_Differential"])
            predictions = lm.predict(transformed_testing)

            print_results(transformed_testing, predictions, 11, reg_flag)
        else:

            training_features["Over_Under"] = numpy.where(
                training_features["Point_Differential"] > training_features["Vegas_Line"], 1, 0)
            testing_features["Over_Under"] = numpy.where(
                testing_features["Point_Differential"] > testing_features["Vegas_Line"], 1, 0)

            model = svm.SVC()
            classifier = model.fit(training_features[feature_columns[7:33]], training_features["Over_Under"])
            predictions = classifier.predict(testing_features[feature_columns[7:33]])
            print_results(testing_features, predictions, 11, reg_flag)


# Create team id lookup
team_dict = create_team_dict()

# File name for raw results
raw_results = "results_data_full.csv"

# Rolling average length
rolling_avg_window = 4

# Hold the columns for the future feature dataframe
feature_columns = ["Week_ID", "Year", "Away_Team", "Away_Score", "Home_Team", "Home_Score", "Favored_Team", "Vegas_Line",
                   "Away_PPG_ma", "Away_FD_ma", "Away_RYPG_ma", "Away_PYPG_ma", "Away_TO_ma",
                   "Away_PPGA_ma", "Away_FDA_ma", "Away_RYPGA_ma", "Away_PYPGA_ma", "Away_TOA_ma", "Away_Win_Rate",
                   "Home_PPG_ma", "Home_FD_ma", "Home_RYPG_ma", "Home_PYPG_ma", "Home_TO_ma",
                   "Home_PPGA_ma", "Home_FDA_ma", "Home_RYPGA_ma", "Home_PYPGA_ma", "Home_TOA_ma", "Home_Win_Rate",
                   "Away_Elo", "Home_Elo", "CF_Result"]

# Initialize years list to pull data for
years = list(map(str, list(range(2002, 2017))))

##########################################################################
# Either crawl web to populate data, or read in .csv containing raw data #
##########################################################################
#web_crawler(years, raw_results, team_dict)
raw_data = pd.read_csv(raw_results)

# Create week_id column
raw_data["Week_ID"] = range(raw_data.shape[0])

# Double check on get_team_results function, should be 16 games * len(years)
for i in range(32):
    assert get_team_results(raw_data, i).shape == (16 * len(years), 17)

# Double check on year_slice function, should be 16 games * 32 teams / 2 for duplicates per year
for i in years:
    assert year_slice(raw_data, int(i), int(i) + 1).shape == (256, 18)

# Create dictionary holding moving average stats for each team
team_results_dict = dict()
for i in range(32):
    temp_results = get_team_results(raw_data, i)
    temp_results["For_PPG"] = temp_results["For_Score"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["For_FD"] = temp_results["For_First_Downs"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["For_RYPG"] = temp_results["For_Rushing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["For_PYPG"] = temp_results["For_Passing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["For_TO"] = temp_results["For_Turnovers"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)

    temp_results["Against_PPG"] = temp_results["Against_Score"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["Against_FD"] = temp_results["Against_First_Downs"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["Against_RYPG"] = temp_results["Against_Rushing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["Against_PYPG"] = temp_results["Against_Passing_Yards"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)
    temp_results["Against_TO"] = temp_results["Against_Turnovers"].rolling(window=rolling_avg_window, min_periods=rolling_avg_window).mean().shift(1)

    temp_results["Win_Rate"] = temp_results["Result"].rolling(window=(rolling_avg_window * 2),
                                                                      min_periods=(rolling_avg_window * 2)).mean().shift(1)

    team_results_dict[i] = temp_results[["Week_ID", "Year", "Week", "Date", "For_Team",
                                         "For_PPG", "For_FD", "For_RYPG", "For_PYPG", "For_TO",
                                         "Against_Team", "Against_PPG", "Against_FD", "Against_RYPG", "Against_PYPG", "Against_TO", "Win_Rate"]]
#print(team_results_dict[0].head())
# Check on results for team_results_dict
for i in range(32):
    assert team_results_dict[i].shape == (16 * len(years), 17)
    assert team_results_dict[i].isnull().sum().sum() == (rolling_avg_window) * 10 + (2 * rolling_avg_window)

#Initialize feature dataframe with moving averages
feature_list = list()
for index, row in raw_data.iterrows():
    list_to_add = list()
    list_to_add.extend(row[["Week_ID", "Year", "Away_Team", "Away_Score", "Home_Team", "Home_Score", "Favored_Team", "Vegas_Line"]])
    away_team_stats = team_results_dict[row["Away_Team"]]
    home_team_stats = team_results_dict[row["Home_Team"]]

    away_stats = away_team_stats.loc[(away_team_stats["Week_ID"] == index), ["For_PPG", "For_FD", "For_RYPG", "For_PYPG", "For_TO",
                                                                          "Against_PPG", "Against_FD", "Against_RYPG",
                                                                          "Against_PYPG", "Against_TO", "Win_Rate"]]
    home_stats = home_team_stats.loc[(home_team_stats["Week_ID"] == index), ["For_PPG", "For_FD", "For_RYPG", "For_PYPG", "For_TO",
                                                                          "Against_PPG", "Against_FD", "Against_RYPG",
                                                                          "Against_PYPG", "Against_TO", "Win_Rate"]]

    for index, row in away_stats.iterrows():
        list_to_add.extend(row)
    for index, row in home_stats.iterrows():
        list_to_add.extend(row)

    feature_list.append(list_to_add)

features = pandas.DataFrame(feature_list, columns=feature_columns[:30])

team_elo_dict = calculate_elo_values(raw_data)

# Check on the number of values in team_elo_dict - should be one for each game, plus one value for after the final week of data
for i in range(32):
    assert len(team_elo_dict[i]) == 16 * len(years) + 1

# Get list of Elo ratings by away/home team
away_elo, home_elo = get_away_home_elos(raw_data, team_elo_dict)

features.loc[:, "Away_Elo"] = away_elo
features.loc[:, "Home_Elo"] = home_elo

#############################################################################
# This marks the beginning of using specific years for training and testing #
#############################################################################

# Get train/test results for each test season from 2007-2016
#performance_by_year(2003, 2013, features, raw_data, feature_columns, False)

training_start_year = 2003
training_final_year = 2006
testing_year = 2007

training_features = year_slice(features, training_start_year, training_final_year + 1)
testing_features = year_slice(features, testing_year, testing_year + 1)

full_results = get_simple_results(raw_data, training_start_year, training_final_year + 1)

model = get_cf_model(full_results)

training_features.loc[:, "CF_Result"] = model.predict(
    [training_features["Away_Team"], training_features["Home_Team"]])
training_features.loc[:, "Point_Differential"] = (
    training_features["Away_Score"].copy() - training_features["Home_Score"].copy())

testing_features.loc[:, "CF_Result"] = model.predict(
    [testing_features["Away_Team"], testing_features["Home_Team"]])
testing_features.loc[:, "Point_Differential"] = (
    testing_features["Away_Score"].copy() - testing_features["Home_Score"].copy())

training_features["Vegas_Line"][training_features["Favored_Team"] == training_features["Away_Team"]] = -1 * \
                                                                                                       training_features[
                                                                                                           "Vegas_Line"][
                                                                                                           training_features[
                                                                                                               "Favored_Team"] ==
                                                                                                           training_features[
                                                                                                               "Away_Team"]]
testing_features["Vegas_Line"][testing_features["Favored_Team"] == testing_features["Away_Team"]] = -1 * \
                                                                                                    testing_features[
                                                                                                        "Vegas_Line"][
                                                                                                        testing_features[
                                                                                                            "Favored_Team"] ==
                                                                                                        testing_features[
                                                                                                            "Away_Team"]]
training_features["Over_Under"] = numpy.where(
                training_features["Point_Differential"] > training_features["Vegas_Line"], 1, 0)
testing_features["Over_Under"] = numpy.where(
                testing_features["Point_Differential"] > testing_features["Vegas_Line"], 1, 0)

model = svm.SVC()
classifier = model.fit(training_features[feature_columns[7:33]], training_features["Over_Under"])
predictions = classifier.predict(testing_features[feature_columns[7:33]])
print_results(testing_features, predictions, 11, False)

training_features.to_csv("training_features_classifier.csv")
testing_features.to_csv("testing_features_classifier.csv")
numpy.savetxt("predictions_classifier.csv", predictions, delimiter=",")
'''model = linear_model.ElasticNet()
lm = model.fit(training_features[feature_columns[8:33]], training_features["Point_Differential"])
predictions = lm.predict(testing_features[feature_columns[8:33]])
print_results(testing_features, predictions, 11)'''


'''training_features.to_csv("training_features.csv")

models = []
models.append(("LR", linear_model.Lasso()))
models.append(("EN", linear_model.ElasticNet()))
models.append(("BR", linear_model.BayesianRidge()))
models.append(("ARD", linear_model.ARDRegression()))
models.append(("Ridge", linear_model.Ridge()))
models.append(("Linear", linear_model.LinearRegression()))
models.append(("SVM", svm.SVR()))
models.append(("SGD", SGDRegressor()))
models.append(("NN", MLPRegressor()))
models.append(("RF", RandomForestRegressor()))
models.append(("GBR", GradientBoostingRegressor()))

results = []
names = []

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
transformed_training = min_max_scaler.fit_transform(training_features[feature_columns[8:33]])
transformed_testing = min_max_scaler.transform(testing_features[feature_columns[8:33]])

for name, model in models:
    lm = model.fit(transformed_training, training_features["Point_Differential"])
    predictions = lm.predict(transformed_testing)
    mse = numpy.mean((predictions - testing_features["Point_Differential"] ** 2))

    score = model.score(transformed_testing, testing_features["Point_Differential"])
    print("%s: %f %f" % (name, score, mse))'''

# Testing for classification model for no cover/cover
training_features["Vegas_Line"][training_features["Favored_Team"] == training_features["Away_Team"]] = -1 * training_features["Vegas_Line"][training_features["Favored_Team"] == training_features["Away_Team"]]
testing_features["Vegas_Line"][testing_features["Favored_Team"] == testing_features["Away_Team"]] = -1 * testing_features["Vegas_Line"][testing_features["Favored_Team"] == testing_features["Away_Team"]]

training_features["Over_Under"] = numpy.where(training_features["Point_Differential"] > training_features["Vegas_Line"], 1, 0)
testing_features["Over_Under"] = numpy.where(testing_features["Point_Differential"] > testing_features["Vegas_Line"], 1, 0)



'''models = []
models.append(("SVM", svm.SVC()))
models.append(("Ada", AdaBoostClassifier()))
models.append(("LR", linear_model.LogisticRegression()))
models.append(("SGD", SGDClassifier(shuffle=True)))
models.append(("NN", MLPClassifier()))
models.append(("RF", RandomForestClassifier()))
models.append(("GBC", GradientBoostingClassifier()))


for name, model in models:
    lm = model.fit(training_features[feature_columns[7:33]], training_features["Over_Under"])
    predictions = lm.predict(testing_features[feature_columns[7:33]])

    score = model.score(testing_features[feature_columns[7:33]], testing_features["Over_Under"])
    print("%s: %f" % (name, score))'''