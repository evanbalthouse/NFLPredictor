import requests
import re
import pandas
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



def moving_average(dataframe, length, indices_home, indices_away):
    column_names = numpy.append(dataframe.columns.values[indices_home], dataframe.columns.values[indices_away])
    column_names_avg = column_names + "_avg"
    column_names_avg = numpy.append(column_names_avg, ["Away_Team", "Home_Team"])
    df = pandas.DataFrame(columns=column_names_avg)
    df["Away_Team"] = dataframe["Away_Team"]
    df["Home_Team"] = dataframe["Home_Team"]

    for column in column_names:
        df[column + "_avg"] = pandas.rolling_mean(dataframe[column], window=length, min_periods=length)

    return df;


def create_game_dictionary(results_data):
    results_data.ix[:, 0:3].to_csv("game_dictionary.csv", header=True, index_label="Game_Id")


def win_flag(x):
    return 1 if x[1] - x[6] > 0 else 0


def write_win_rate(team_file_stump):
    for i in range(0, 32):
        team_data = pandas.read_csv(team_file_stump + str(i) + ".csv")

        for j in [team_data]:
            j["Win_Flag"] = team_data.apply(win_flag, axis=1)
            win_rate = list()
            k = 0
            for index, row in team_data.iterrows():
                if k < 16:
                    win_rate.append(-1)
                else:
                    win_rate.append(sum(team_data["Win_Flag"].iloc[(k - 16):k]) / 16)

                k += 1
            j["Win_Rate"] = win_rate
            del j["Win_Flag"]
        team_data.to_csv("team_data/raw_data_win_rate_" + str(i) + ".csv", index=False)


def write_team_stats(results_data):
    for i in range(0, 1):
        team_results = pandas.concat([results_data.loc[results_data["Away_Team"] == i], results_data.loc[results_data["Home_Team"] == i]]).sort()
        writer = csv.writer(open(directory_stump + "team_data/raw_data_" + str(i) + ".csv", "w", newline=""))
        writer.writerow(["Game_Id", "Score_for", "First_Downs_for", "Rushing_Yards_for", "Passing_Yards_for", "Turnovers_for", "Score_against", "First_Downs_against", "Rushing_Yards_against", "Passing_Yards_against", "Turnovers_against", "Home_flag", "Opponent"])
        for index, row in team_results.iterrows():
            to_write = list()
            to_write.append(index)
            for_string = "Away"
            against_string = "Home"
            home_flag = 0
            opponent = row["Home_Team"]

            if i == row["Home_Team"]:
                for_string = "Home"
                against_string = "Away"
                home_flag = 1
                opponent = row["Away_Team"]

            to_write.append(row[for_string + "_Score"])
            to_write.append(row[for_string + "_First_Downs"])
            to_write.append(row[for_string + "_Rushing_Yards"])
            to_write.append(row[for_string + "_Passing_Yards"])
            to_write.append(row[for_string + "_Turnovers"])
            to_write.append(row[against_string + "_Score"])
            to_write.append(row[against_string + "_First_Downs"])
            to_write.append(row[against_string + "_Rushing_Yards"])
            to_write.append(row[against_string + "_Passing_Yards"])
            to_write.append(row[against_string + "_Turnovers"])
            to_write.append(home_flag)
            to_write.append(opponent)

            writer.writerow(to_write)


def write_elo_values(team_file_stump, results_data):
    elo_matrix = list()
    for i in range(0, 32):
        team_elo = list()
        team_elo.append(1500)
        elo_matrix.append(team_elo)

    for index, row in results_data.iterrows():
        away_team = row["Away_Team"]
        home_team = row["Home_Team"]
        away_score = row["Away_Score"]
        home_score = row["Home_Score"]

        res = 1
        if away_score < home_score:
            res = 0
        elif away_score == home_score:
            res = 0.5

        away_elo_series = elo_matrix[away_team]
        home_elo_series = elo_matrix[home_team]

        away_elo_prev = away_elo_series[len(away_elo_series) - 1]
        home_elo_prev = home_elo_series[len(home_elo_series) - 1]

        y_away = 0
        y_home = 0

        if row["Week"] == 1 and row["Year"] != 2002:
            y_away = 0.5 * (1500 - away_elo_prev)
            y_home = 0.5 * (1500 - home_elo_prev)

        away_elo_prev += y_away
        home_elo_prev += y_home

        m_away = numpy.log(abs(away_score - home_score) + 1) * (2.2 / (0.001 * (away_elo_prev - home_elo_prev) + 2.2))
        m_home = numpy.log(abs(home_score - away_score) + 1) * (2.2 / (0.001 * (home_elo_prev - away_elo_prev) + 2.2))

        express_away = 1 / (1 + 10 ** ((home_elo_prev - home_elo_prev) / 400))
        express_home = 1 / (1 + 10 ** ((home_elo_prev - away_elo_prev) / 400))

        away_elo_new = away_elo_prev + 20 * m_away * (res - express_away)
        home_elo_new = home_elo_prev + 20 * m_home * ((1 - res) - express_home)

        elo_matrix[away_team].append(away_elo_new)
        elo_matrix[home_team].append(home_elo_new)

    for i in range(0, 32):
        team_data = pandas.read_csv(team_file_stump + "win_rate_" + str(i) + ".csv")
        team_data["ELO_Rating"] = elo_matrix[i][1:]
        team_data.to_csv(directory_stump + "team_data/raw_data_elo_" + str(i) + ".csv", index=False)


def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype="int64", name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)


def create_bias(inp, n_in):
    x = Embedding(n_in, 1, input_length=1)(inp)
    return Flatten()(x)
