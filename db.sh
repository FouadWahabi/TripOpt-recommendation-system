#!/bin/bash

if pgrep -x "mongod" > /dev/null
then
    echo "Restoring the TripOpt db"
    sudo mongorestore --db trip_opt --drop trip_opt_db
else
    echo "MognoDB is not running please run it before loading the data."
fi
