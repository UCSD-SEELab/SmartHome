import sys
sys.path.append('../')

from utils.preliminaries import *
from utils.utils import *


date_of_data_collected = "05-14-2018"

filename = '../../data/' + date_of_data_collected + '/videos/recording.mp4'

ambulatory_activities = {1: "lying", \
						 2: "sitting", \
						 3: "standing", \
						 4: "walking", \
						 5: "nothing"}

kitchen_activities = {1: "nothing", \
					  2: "food pred", \
					  3: "stove", \
					  4: "microwave", \
					  5: "oven", \
					  6: "setting table", \
					  7: "serving food", \
					  8: "eating and drinking", \
					  9: "making coffee", \
					  10: "clearing table", \
					  11: "cleaning table", \
					  12: "washing dishes by hand", \
					  13: "washing dishes by dishwasher", \
					  14: "putting away dishes", \
					  15: "cleaning floor"}

f = open("../../data/" + date_of_data_collected + "/labels_test.txt","w+")

# Read video
cap = cv2.VideoCapture(filename)
# The window size for labeling data (in second)
window_size = 3.0
# Get frame rate
fps =  cap.get(5)
# Convert frame rate to miniseconds
speed = int( (1/(fps+0.0)*1000))

# The number of frames in the window
max_cnt =  int(window_size/(1/(fps+0.0)))

start_date = "2018-05-14 14:03:09"
delta = datetime.timedelta(seconds = window_size)
start_date = parser.parse(start_date)

# Count how many frames have been displayed
cnt = 0

while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame', gray)

	# Display at normal speed
	if cv2.waitKey(speed) & 0xFF == ord('q'): 
	    break

	cnt = cnt + 1
	if cnt == max_cnt:
		print "########################################################################"
		print ambulatory_activities
		print colored('Choice?','red')

		index = checkInput()
		while ambulatory_activities.has_key(int(index)) == False:
			print colored('Wrong choice, please input again.','red')
			index = checkInput()
		basic_activity = ambulatory_activities[int(index)]

		for index , (key, value) in  enumerate(kitchen_activities.items()):
			if index % 4 == 0:
				print "\n"
			print key, value + "	",

		print colored('Choice?','red')
		index = checkInput()
		while kitchen_activities.has_key(int(index)) == False:
			print colored('Wrong choice, please input again.','red')
			index = checkInput()
			
		kitchen_activity = kitchen_activities[int(index)]
		start_date = start_date + delta
		f.write(start_date.strftime("%Y-%m-%d %H:%M:%S") + " " + basic_activity + " " + kitchen_activity + "\n")
		cnt = 0

f.close()
cap.release()
cv2.destroyAllWindows()