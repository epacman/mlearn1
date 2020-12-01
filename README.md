# mlearn1
Machine learning program for predicting direction of OMXS30 index between 12:00 and 17:30. mlearn_experimental.py is the main file. Current accuracy as of 2020-12-01 is around 56 %

Ideas to test:
* Worse accuracy on tuesdays and fridays, at least during most of 2020 (tested in ProRealtime) Can this be taken into account?
* Model could possibly be helped by checking if price is above or below some moving averages at 12:00. Confirmed on MA(160) in 30 min timeframe in ProRealtime. Test!
* 
