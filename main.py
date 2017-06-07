from __future__ import division
from __future__ import print_function

import time
import sys
from pylab import *
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from bson.objectid import ObjectId


def _connect_mongo(host, db, port=27017, username=None, password=None):
    """ Creating a connection with a Mongo database """
    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def load_train_set(db):
    # Like value
    like = 0.002
    """ Result train_set """
    train_set = {}

    """ Load the list of users from the db"""
    users = db['user'].find()

    """ Load the list of activities from the db """
    activities_categories = db['CategoryActivity'].find()
    activities = {}
    categories = set([])
    for activity_category in activities_categories:
        categories.add(activity_category['categoryId'])
        if not activities.get(activity_category['activityId']):
            activities[activity_category['activityId']] = [activity_category['categoryId']]
        else:
            activities[activity_category['activityId']].append(activity_category['categoryId'])

    """ Load the list of votes from the db"""
    votes = db['Vote'].find()
    for vote in votes:
        user_index = 0
        if not train_set.get(vote['userId']):
            train_set[vote['userId']] = {}

        activity_categories = activities[vote['activityId']]
        value = like if bool(vote['value']) else -like
        for activity_category in activity_categories:
            if not train_set[vote['userId']].get(activity_category):
                train_set[vote['userId']][activity_category] = 0

            train_set[vote['userId']][activity_category] += value

    print(train_set)


db = _connect_mongo("localhost", "trip_opt")
load_train_set(db)
