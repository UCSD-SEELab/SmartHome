This is an export of data for a single Subject within your Study from
the MC10 Cloud. This zip file contains all Recordings and Annotations
uploaded during the Study for that Subject up to the point the export
was requested. This data is contained in CSV files grouped into
various subfolders as described below.

# Folder Structure Explained

## <studyName>/<subjectName>/<sensorLocation>/<sensorId>/<recording-start>/<recordingCsvFile>

Where
  - studyName: name of the study
  - subjectName: name of the subject
  - sensorLocation: body location of the sensor when the recording was
  made.
  - sensorId: id of the sensor
  - recordingStart: time when the recording started

e.g. my_study/william_smith/bicep_right/d417785d4191/2016-09-22T16-05-03-850Z/accel.csv

Note: time is format as YYYY-MM-DDTHH-MM-SS-TTTZ where TTT is thousands of a second. All times in GMT.

## <studyName>/<subjectName>/annotations.csv

Optional "annotations.csv" file if Subject completed Activities or Surveys.

### CSV File Formats:

#### gyro.csv
- Gyroscope Output
- Time in GMT Unix Epoch microseconds
- Gyroscope values in degrees per second
- Header row "Timestamp (microseconds),Gyro X (°/s),Gyro Y (°/s),Gyro Z (°/s)"
- Example data: "1458217123808462,2.32433211233211193,1.34474969474969441,1.031379731379734"

#### accel.csv
- Accelerometer values
- Time in GMT Unix Epoch microseconds
- Acceleration values in G's
- Header row "Timestamp (microseconds),Accel X (g),Accel Y (g),Accel Z (g)"
- Example data: "1458217123808462,0.11233211233211193,0.09474969474969441,1.1731379731379734"

#### elec.csv
- Electrode values
- Time in GMT Unix Epoch microseconds
- Voltages values in Volts
- Header row "Timestamp (microseconds), Sample (V)"
- Example data: "1458217123808462,0.0710072226036978"

### XXXX-errors.csv
- may be accel-errors.csv, gyro-errors.csv, elec-errors.csv
- Time in GMT Unix Epoch microseconds
- Header row "Timestamp (microseconds),Reason"
- Example data: "1482329518825214,EMPTY_SAMPLE"

#### annotations.csv
- Annotation information from Subject completed Activities or Diaries
- Time in GMT Unix Epoch microseconds
- Header row "Timestamp (ms),AnnotationId,EventType,AuthorId,Start Timestamp (ms),Stop Timestamp (ms),Value"
- Example data: "1456433577218,bf985e20-dc01-11e5-b6f7-0a2e9e9dd655,DiaryQuestion:Yes or No,6e3047c1-1a4d-4f00-a8e2-7a1f5b2e2dc4,1456433576921,1456433576921,YES"
- AnnotationId: UUID uniquely identifying the Annotation
- EventType: concatenation of "<Activity|Dairy>" or "<Activity|Diary>Question:<Question>"
- Start/Stop Timestamp: The respective beginning and end of the event
- AuthorId: UUID of User logged into app that completed the Activity/Diary

## Plotting Examples

See github.com/MC10Inc/biostamp_rc for plotting examples of all data types.
