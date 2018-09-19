# TripOpt recommendation engine

This is the code source of the recommendation system behind TripOpt: which is travel recommendation app that uses Machine learning to learn from the user’s interactions and recommend a tailored custom itinerary to follow in a certain city: tripopt.com

## Get the glimpse behind the curtain

Now I am going to explain the idea behind the recommendation algorithm that makes the power of TripOpt as a travel recommendation app, and that ensures the accuracy of the recommendation so that the end user get the best results each time he asks for it.

### History

#### Content-based

Let's start with a brief history of the algorithm and how did we get to the "Perfect" algorithm.

In the first alpha versions of TripOpt we started we a simple `content-based` recommendation algorithm based on the category of the activities and the likes/dislikes of the user.

So the idea is :

 - Each activity is represented by a vector of 1/0, where the columns are the list of the categories that we have in the database. So if the value is 0 for category1 then this activity doesn't has this category, otherwise the value is 1.
 - Each user is represented by a vector of weights: the weight of the category toward this user, so if the user likes a lot of activities with specific category1, then the weight of category1 will grow.
 - To get the result of the recommendation we just multiply the two matrices: the activities matrix with the weights matrix and then we sort the activities according to the result.

Below is an example :

Assume that we have three activities :

Carpe Diem : (nightlife)
Bardo museum : (historical, culture)
Soltan l bihar : (culture, restaurant, food)

So the activies matrix will be :

                   night.   histo. cult.  rest.  food.
<img src="https://imgur.com/HIOAE1O" style="width: 400px;"/>

Now Assuming that the weights matrix is like this :

<img src="https://imgur.com/L8N6WkN" style="width: 400px;"/>

So computing the result now :

<img src="https://imgur.com/nYQPL1E" style="width: 100px;"/>

And here we go we got the recommendation score of each Activity for each User.

**BUT**, why we get rid of this method despite that we got good results :

  - Performance : this method is very greedy as the matrix size grows linearly as the number of users*activities grows, so this will take a long delay to compute the recommendation.

  - Not really a good result : because we need also to show some activities that the user didn't know about and he may likes it.


#### Collaborative filtering

After trying the previous method with a bunch of a sample users, we discovered the importance of showing to the user activities that are not in the list of his interests but that he has a big probability that he likes it. So we opt to the collaborative filtering method.

For this method we used PredictionIO, which was really *awful* !!

**So**, the results wasn't as expected :

 - The result is very sensitive to the training, so any wrong like/dislike could affect the result.
 - We encountered a big problem with the filtering, as have the feature of the possible to filter the recommendation based on specific list of categories, specific city, specific country, distance etc... , So we are obliged to take in consideration the activity informations.

*Conclusion*: Let's **go back** to the content-based recommendation :'( .

> "Hey! Hey ! Wait, I have an **idea**.", I said.

## Clustered content based recommendation

> "What the hell is this ?." My team mates said

The idea behind the new algorithm is to go back to the first method (content-based), but in this case we will add a new clustering layer, so that we group the users based on theirs weights and we create a *representative*  for each group who has the most similarities to all the group users.

What we gained from this :

 1. Performance: We used the `elbow method` to compute the best number of groups, which is too much less than the number of the users. Approximately :  ![](https://qph.ec.quoracdn.net/main-qimg-6a0c11e1bce3a5460b762a4c5ca1ff69)
 2. Variety of recommended activities: As the representative has some difference with each user of the group, we will get some activities that are not in the list of interest of the user but probably will like these activities.

## Built With

This algorithm is implemented using `Python` programming language, using the [`tensorflow`](https://www.tensorflow.org/) as Machine Learning library and [`Flask`](flask.pocoo.org/) as a web framework to create the REST endpoints.

## Contributing

Thanks for thinking in contributing to this project. Please email me with any suggestions: [Dear Fouad,](mailto:fouad.wahabi@gmail.com)


## Authors

* **Fouad Wahabi** - Computer Guru  - [LinkedIn](https://www.linkedin.com/in/wfouad) - [Email](mailto:fouad.wahabi@gmail.com) - [Website](wfouad.com)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

