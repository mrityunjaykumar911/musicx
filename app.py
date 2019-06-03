from __future__ import absolute_import
import socket
import json
import os
import numpy as np
import pickle
from tqdm import tqdm
import sys
import ast
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

from flask import Flask, render_template, request, redirect, Response, jsonify, send_file
import pandas as pd

# First of all you have to import it from the flask module:
from sklearn import metrics, manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from logic import get_similar_moods_calc

MY_STATIC_LIST = ["reflective", "pleasant", "powerful"]

# suppress deprecated warnings
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__,
            static_url_path='/static',
            template_folder='www')

################# ONLY MUSIC API
df_tracks = None
modified_tags = {}
df_tags = None
df_ma = None
all_activities_static = {
	"activity.daily_routines": "rdroutine",
	"activity.emotional": "remotional",
	"activity.entertainment": "rentertain",
	"activity.intellectual": "rintellec",
	"activity.live_music": "rlmusic",
	"activity.music_listening": "rmlisten",
	"activity.on_the_move": "ronTheMove",
	"activity.physical": "rphysical",
	"activity.social": "rsocial"}

all_mood_list_static = None
mood_vs_mood = None
mood_vs_acts = None
all_activities_static_sorted = ['daily_routines', 'emotional', 'entertainment',
                                'intellectual', 'live_music', 'music_listening',
                                'on_the_move', 'physical', 'social']

SPOTIPY_CLIENT_ID = '46cde2c012e444fbbdd451b5d6adfad4'
SPOTIPY_CLIENT_SECRET = '87d073dcb76b43778dcc39c9192751d0'

client_id = SPOTIPY_CLIENT_ID
client_secret = SPOTIPY_CLIENT_SECRET
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_track_infomation(track_id):
	answer_dict = {}
	result_ret = df_tracks.loc[df_tracks["trackid"] == track_id]
	# print(result_ret)
	if result_ret.trackid.count() != 0:
		each = result_ret.to_dict()
		index = list(each["artistname"].keys()).pop()
		answer_dict["artistname"] = each["artistname"][index]
		answer_dict["tracktitle"] = each["tracktitle"][index]
		answer_dict["releasetitle"] = each["releasetitle"][index]
		xxx__ = None
		try:
			xxx__ = ast.literal_eval(each["spotify"][index])
			xxx__["external_urls"]["spotify"] = xxx__["external_urls"]["spotify"].replace("/track", "/embed/track")
			answer_dict["spotify"] = xxx__
		except:
			xxx__ = {}
	
	return answer_dict


def cluster_query(mood_list, activity_list):
	ans_moods = []
	ans_acts = []
	ans = []
	for tr, vals in modified_tags.items():
		_m_relv = vals["mood_relev"]
		_a_relv = vals["act_relev"]
		# print("m_relv",_m_relv)
		# print("_a_relv",_a_relv)
		matched_m = []
		for m in mood_list:
			if m in _m_relv.keys():
				_m_imp_score_ = _m_relv[m]
				matched_m.append((m, _m_imp_score_))
		matched_a = []
		for a in activity_list:
			if a in _a_relv.keys():
				_a_imp_score_ = _a_relv[a]
				matched_a.append((a, _a_imp_score_))
		if len(matched_m) != 0 or len(matched_a) != 0:
			ans.append((tr, matched_m, matched_a))
	#     print(len(ans))
	relevancy_scores = get_cluster_scores(ans, mood_list, activity_list)
	return relevancy_scores


def flat_query(mood_list, activity_list, topk=10):
	ans_moods = []
	ans_acts = []
	ans = []
	for tr, vals in modified_tags.items():
		_m_relv = vals["mood_relev"]
		_a_relv = vals["act_relev"]
		# print("m_relv",_m_relv)
		# print("_a_relv",_a_relv)
		matched_m = []
		for m in mood_list:
			if m in _m_relv.keys():
				_m_imp_score_ = _m_relv[m]
				matched_m.append((m, _m_imp_score_))
		matched_a = []
		for a in activity_list:
			if a in _a_relv.keys():
				_a_imp_score_ = _a_relv[a]
				matched_a.append((a, _a_imp_score_))
		if len(matched_m) != 0 or len(matched_a) != 0:
			ans.append((tr, matched_m, matched_a))
	#     print(len(ans))
	relevancy_scores = get_relevancy_score(ans, mood_list, activity_list, top_k=topk)
	return relevancy_scores


def get_max_mood_len():
	all_columns_tags = df_tags.columns.to_list()
	max_mood_length = len([xx for xx in all_columns_tags if 'mood' in xx])
	return max_mood_length


def get_max_acts_len():
	all_columns_tags = df_tags.columns.to_list()
	max_acts_length = len([xx for xx in all_columns_tags if 'activity' in xx])
	return max_acts_length


def get_activity_vector_for_mood(input_mood):
	"""

    :param input_mood:
    :return:
    get_activity_vector_for_mood(sad)

    [16.0, 110.0, 21.0, 29.0, 24.0, 57.0, 25.0, 21.0, 21.0]
    """
	xxx = mood_vs_acts.get(input_mood, "")
	return list(map(lambda x: x, sorted(xxx.items(), key=lambda x: x[0])))


def get_cluster_scores(answer, mood_list, activity_list, top_k=10):
	max_score = len(mood_list) + len(activity_list)
	if max_score == 0.0:
		return None
	
	def filter_function(xx_):
		return len(xx_[1]) != 0 and len(xx_[2]) != 0
	
	most_rel_first_list = list(filter(lambda x: filter_function(x), answer))
	rest_list = list(filter(lambda x: not filter_function(x), answer))
	xxx = list(map(lambda y: (y[0], y[1:][0]), list(map(lambda x: (x[0], x[1] + x[2]), most_rel_first_list))))
	
	def summation(list_vals):
		return sum(list(map(lambda x: x[-1], list_vals)))
	
	xxx = sorted(list(map(lambda x: (x[0], summation(x[1:][0])), xxx)), key=lambda x: x[1], reverse=True)
	
	data_ = list(map(lambda x: x[-1], xxx))
	no_of_bins = 10
	bin_size = np.ceil((max(data_) - min(data_)) / no_of_bins)
	
	final_answer = {}
	for track_id, val in xxx:
		yyy = int((val - min(data_)) / bin_size)
		
		prev = final_answer.get(yyy, [])
		prev.extend([(track_id, val)])
		final_answer[yyy] = prev
	return final_answer


def get_relevancy_score(answer, mood_list, activity_list, top_k=10):
	max_score = len(mood_list) + len(activity_list)
	if max_score == 0.0:
		return None
	final_ans = []
	max_mood_length = get_max_mood_len()
	max_acts_length = get_max_acts_len()
	
	def filter_function(xx_):
		return len(xx_[1]) != 0 and len(xx_[2]) != 0
	
	most_rel_first_list = list(filter(lambda x: filter_function(x), answer))
	rest_list = list(filter(lambda x: not filter_function(x), answer))
	
	def get_score(input_answer_list):
		ans = []
		for each_track, each_extracted_emo_list, each_act_list in tqdm(input_answer_list, ascii=True,
		                                                               desc="Processing"):
			sum_emo = sum(list(map(lambda x: x[1], each_extracted_emo_list)))
			sum_acts = sum(list(map(lambda x: x[1], each_act_list)))
			track_info = get_track_infomation(each_track)
			ans.append((each_track, (sum_emo + sum_acts) / (max_acts_length + max_mood_length), each_extracted_emo_list,
			            each_act_list, track_info))
		ans = sorted(ans, key=lambda x: x[1], reverse=True)
		ans = list(map(lambda x: x[-1], ans))
		return ans
	
	final_ans.extend(get_score(most_rel_first_list))
	if top_k > len(most_rel_first_list):
		final_ans.extend(get_score(rest_list))
	
	return final_ans[:top_k]


def get_relevant_moods(one_entry):
	ans = {}
	for key in one_entry.keys():
		if "mood" in key:
			_v_ = one_entry[key]
			if _v_ >= 1.0:
				ans[key[5:]] = _v_
	return ans


def get_relevant_activity(one_entry):
	ans = {}
	for key in one_entry.keys():
		if "activity" in key:
			v__ = one_entry[key]
			if v__ >= 1.0:
				ans[key[9:]] = v__
	return ans


# @app.route("/get_mds_data")
def get_data_for_mds():
	final_data = []
	_vals_15 = df_ma["mood"].tolist()[:]
	some = df_ma[df_ma["mood"].isin(_vals_15)]
	ma_mods = some[pd.isnull(some).any(axis=1) == False]
	ma_mods.reset_index(inplace=True)
	del ma_mods["index"]
	X = ma_mods.values[:, 3:]
	# embedding = MDS(n_components=2, random_state=1, n_jobs=-1, max_iter=30)
	X = X[:100]
	
	dm1 = pairwise_distances(X, metric='euclidean')
	mds1 = MDS(n_components=2, dissimilarity='precomputed')
	X_transformed = mds1.fit_transform(dm1)
	for i in range(0, len(X_transformed)):
		single_data = dict({"x": X_transformed[i, 0], "y": X_transformed[i, 1]})
		final_data.append(single_data)
	return jsonify({"data": final_data})


def pre_process():
	global df_ma, df_tags, df_tracks, modified_tags, mood_vs_mood, mood_vs_acts, all_mood_list_static
	df_tracks = pd.read_csv("data/tracks22.csv")  # searches.csv
	df_tags = pd.read_csv("data/tags.csv")
	df_ma = pd.read_csv("data/moodactivity2.csv")
	all_mood_list_static = df_ma.mood.unique().tolist()
	all_columns_tags = df_tags.columns.to_list()
	
	if os.path.exists("mod.m"):
		modified_tags = pickle.load(open("mod.m", "rb"))
		print("pickle loaded done")
	else:
		combined_dict = {}
		dummy_dict = {k: None for k in all_columns_tags}
		for index, row in tqdm(df_tags.iterrows()):
			trackid = row["trackid"]
			dummy_dict_copy = combined_dict.get(row["trackid"], dummy_dict.copy())
			for key in all_columns_tags:
				xxx = row[key]
				if key == "userid":
					prev = dummy_dict_copy.get(key, None)
					if prev is None:
						prev = []
					prev.append(xxx)
				else:
					prev = dummy_dict_copy.get(key, 0.0)
					if prev == None:
						prev = 0.0
					if not xxx >= 0.0:
						xxx = 0.0
					prev += xxx
				dummy_dict_copy[key] = prev
			combined_dict[trackid] = dummy_dict_copy
		
		for track_id, v in tqdm(combined_dict.items()):
			mood_result = get_relevant_moods(v)
			acts_result = get_relevant_activity(v)
			v_copy = v.copy()
			v_copy["mood_relev"] = mood_result
			v_copy["act_relev"] = acts_result
			modified_tags[track_id] = v_copy
		pickle.dump(modified_tags, open("mod.m", "wb"))
		print("pickle done")
	
	mood_vs_mood = pickle.load(open("data/final_dict.m", "rb"))
	mood_vs_acts = pickle.load(open("data/cm_mood_act.m", "rb"))


#################
@app.route("/get_mds_data", methods=["GET"])
def get_mds():
	ans = get_data_for_mds()
	return ans


@app.route("/get_all_moods", methods=["GET"])
def get_all_moods():
	return jsonify({"data": all_mood_list_static})


@app.route("/get_kde_param_opp", methods=["POST"])
def get_kde_param_opp():
	content = request.get_json()
	input_mood = content.get("mood", [])
	vectors1 = get_activity_vector_for_mood(input_mood=input_mood)
	# vectors = [["activity %d" % (i+1),j] for i,j in enumerate(ans)]
	most_dissimilar = get_dissimilar_moods_func(all_moods_posted=[input_mood], limit=1)
	opp_mood = most_dissimilar.get_json().get("data")[0][0]
	vectors2 = get_activity_vector_for_mood(opp_mood)
	combined = [(k[0][0], k[0][1], k[1][1]) for k in zip(vectors1, vectors2)]
	return jsonify({"data": combined, "mood_1": input_mood, "mood_2": opp_mood})


@app.route("/get_kde_param", methods=["POST"])
def get_kde_params():
	content = request.get_json()
	input_mood = content.get("mood", [])
	vectors = get_activity_vector_for_mood(input_mood=input_mood)
	# vectors = [["activity %d" % (i+1),j] for i,j in enumerate(ans)]
	return jsonify({"data": vectors})


def get_rand_activities():
	global df_ma
	activities = sorted([i for i in df_ma.columns.tolist() if 'act' in i])
	random_indexes = np.random.random_integers(len(activities) - 1, size=(1, 3)).tolist().pop()
	return list(
		map(lambda x: x[1], list(filter(lambda x: x[0] in random_indexes, enumerate(all_activities_static.values())))))


@app.route("/get_rose_data2", methods=["POST"])
def get_rose_data2():
	limit = 10
	content = request.get_json()
	limit = int(limit)
	all_moods_posted = content.get("moods", [])
	
	if all_moods_posted[0] == "":
		return jsonify({"data": []})
	combined = {}
	for each_m in all_moods_posted:
		try:
			v = sorted(mood_vs_mood[each_m].items(), key=lambda x: x[1], reverse=True)[:limit]
			combined[each_m] = v
		except Exception as ex:
			print(ex)
	combined = get_boosted_combines_moods(combined)
	combined = list(filter(lambda x: x[0] not in all_moods_posted, combined))
	answers = combined[:limit]
	vectors = []
	for (m_, each_input_mood) in answers:
		xxx = {}
		xxx ["angle"] = m_
		act_vector = get_activity_vector_for_mood(m_)
		xxx["total"] = sum(j for i, j in act_vector)
		for i, j in act_vector:
			xxx[i.replace("activity.", "")] = j
		vectors.append(xxx)

	# vectors = [{**{"angle": m_}, **{i.replace("activity.", ""): j for i, j in get_activity_vector_for_mood(m_)},
	#             **{"total": sum(j for i, j in get_activity_vector_for_mood(m_))}}
	#            for (m_, each_input_mood) in answers]
	
	return jsonify({"data": vectors, "acts": all_activities_static_sorted})


@app.route("/get_rose_data", methods=["POST"])
def get_rose_data():
	limit = 10
	try:
		content = request.get_json()
		limit = int(limit)
		all_moods_posted = content.get("moods", [])
		input_mood = content.get("mood", [])
		if len(input_mood) == 0:
			all_moods_posted = ["happy"]
	except Exception as ex:
		all_moods_posted = ["happy"]
	
	if all_moods_posted[0] == "":
		return jsonify({"data": []})
	combined = {}
	for each_m in all_moods_posted:
		try:
			v = sorted(mood_vs_mood[each_m].items(), key=lambda x: x[1], reverse=True)[:limit]
			combined[each_m] = v
		except Exception as ex:
			print(ex)
	combined = get_boosted_combines_moods(combined)
	combined = list(filter(lambda x: x[0] not in all_moods_posted, combined))
	answers = combined[:limit]
	
	vectors = [{"mood": m_, "vector": get_activity_vector_for_mood(input_mood=m_)} for (m_, each_input_mood) in answers]
	
	return jsonify({"data": vectors})


@app.route("/get_default_moods")
def get_def_moods():
	if all_mood_list_static is None:
		moods = MY_STATIC_LIST
	else:
		try:
			random_indexes = np.random.random_integers(len(all_mood_list_static), size=(1, 4)).tolist().pop()
			moods = [all_mood_list_static[i] for i in random_indexes]
		except:
			moods = MY_STATIC_LIST
	xxx = json.dumps({"def_labels": moods, "def_activities": get_rand_activities()})
	return jsonify({"data": xxx})


def get_boosted_combines_moods(combined_moods, reverse=True):
	ans = {}
	for k, v in combined_moods.items():
		for each in v:
			key = each[0]
			value = each[1]
			prev = ans.get(key, 0.0)
			prev += value
			ans[key] = prev
		max_score = max(map(lambda x: x[1], ans.items()))
		min_score = min(map(lambda x: x[1], ans.items()))
		final_answer = {}
		scaled = 100
		for k, v in ans.items():
			final_answer[k] = (((v - min_score) / max_score) * scaled) + 10.0
		return sorted(final_answer.items(), key=lambda x: x[1], reverse=reverse)


@app.route("/get_similar_moods/<limit>", methods=["POST"])
def get_similar_moods(limit=10):
	print(request.is_json)
	limit = int(limit)
	content = request.get_json()
	all_moods_posted = content.get("moods", [])
	if all_moods_posted[0] == "":
		return jsonify({"data": []})
	combined = {}
	for each_m in all_moods_posted:
		try:
			v = sorted(mood_vs_mood[each_m].items(), key=lambda x: x[1], reverse=True)[:limit]
			combined[each_m] = v
		except Exception as ex:
			print(ex)
	combined = get_boosted_combines_moods(combined)
	combined = list(filter(lambda x: x[0] not in all_moods_posted, combined))
	answers = combined[:limit]
	print(answers)
	return jsonify({"data": answers})


def get_dissimilar_moods_func(all_moods_posted, limit=10):
	combined = {}
	for each_m in all_moods_posted:
		v = sorted(mood_vs_mood[each_m].items(), key=lambda x: x[1], reverse=True)[:limit]
		combined[each_m] = v
	combined = get_boosted_combines_moods(combined, reverse=False)
	answers = combined[:limit]
	return jsonify({"data": answers})


@app.route("/get_opposite_moods/<limit>", methods=["POST"])
def get_dissimilar_moods(limit=10):
	print(request.is_json)
	limit = int(limit)
	content = request.get_json()
	all_moods_posted = content.get("moods", [])
	combined = {}
	for each_m in all_moods_posted:
		v = sorted(mood_vs_mood[each_m].items(), key=lambda x: x[1], reverse=True)[:limit]
		combined[each_m] = v
	combined = get_boosted_combines_moods(combined, reverse=False)
	answers = combined[:limit]
	return jsonify({"data": answers})


@app.route("/user_callback/")
def user_page():
	return render_template("user.html")


@app.route("/user.html")
def tsk0():
	return render_template("user.html")


def fill_track_names(ans):
	answer = []
	for dict_temp in tqdm(ans):
		temp = dict_temp.copy()
		tr_info = get_track_infomation(temp["trackid"])
		temp["name"] = tr_info.get("tracktitle", "")
		answer.append(temp)
	return answer


def finetune_links(ans):
	answer = []
	for i, dict_temp in enumerate(ans):
		temp = dict_temp.copy()
		temp["index"] = i + 1
		answer.append(temp)
	
	for each in answer:
		print(each)
	print(answer)


import operator


def find_source(values):
	index, value = max(enumerate(list(map(lambda x: x[1], values))), key=operator.itemgetter(1))
	return index, value


def make_source_dest_link(ans):
	# print(ans)
	answer = []
	index = 0
	for group_id, values in ans.items():
		source = find_source(values)
		target_index = source[0] + index
		answer.append({"source": 0, "target": index, "value": np.log(source[1])})
		if len(values) == 1:
			answer.append({"source": index, "target": target_index, "value": np.log(source[1])})
			index += 1
			continue
		for i, each in enumerate(values):
			if i != source[0]:
				answer.append({"source": index, "target": target_index, "value": np.log(each[1])})
			index += 1
	return answer


def decorate_nodes(ans):
	answer = []
	dummy_dct = {"trackid": None, "group": None, "name": None, "link_length": None}
	
	xx = dummy_dct.copy()
	xx["trackid"] = "Center"
	xx["link_length"] = 0
	xx["group"] = 0
	xx["name"] = "Center"
	answer.append(xx)
	
	for k, each in ans.items():
		for m in each:
			xxx = dummy_dct.copy()
			xxx["trackid"] = m[0]
			xxx["link_length"] = m[1]
			xxx["group"] = k
			answer.append(xxx)
	
	links = make_source_dest_link(ans)
	
	return {"nodes": answer, "links": links}


@app.route("/cquery", methods=["POST"])
def get_get_canswer():
	data = request.get_json()
	ans = cluster_query(mood_list=data["mood_list"],
	                    activity_list=data["activity_list"])
	ans = decorate_nodes(ans)
	ans["nodes"] = fill_track_names(ans["nodes"])
	ans["nodes"] = list(map(lambda x: {"group": x["group"], "name": x["name"]}, ans["nodes"]))
	return jsonify({"result": ans})


@app.route("/query/<limit>", methods=["POST"])
def get_get_answer(limit=10):
	data = request.get_json()
	limit = int(limit)
	print(data, limit)
	ans = flat_query(mood_list=data["mood_list"],
	                 activity_list=data["activity_list"], topk=limit)
	return jsonify({"result": ans})


@app.route("/kde2")
def draw_temp_kde2():
	return render_template("d3_kerden.html")


@app.route("/kde")
def draw_temp_kde():
	return render_template("dmo.html")


@app.route("/rose")
def rose():
	return render_template("rose.html")


@app.route("/")
def default():
	return render_template("index.html")


@app.route("/admin")
def open_admin():
	return render_template("admin.html")


@app.route("/data.csv")
def csv_get():
	return send_file("www/data.csv")


pre_process()
if __name__ == "__main__":
	pre_process()
	port = 5600  # get_random_port()
	app.run(host="0.0.0.0", port=port, debug=True)
