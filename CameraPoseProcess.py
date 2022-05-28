import os
import csv

class CameraPoseProcess():
    def __init__(self):
        pass

    @staticmethod
    def process_camera_poses(pose_path):
        cleaned_pose_path = pose_path[:pose_path.rfind(".csv")] + "_cleaned.csv"

        header_rows = []
        rows = []

        with open(pose_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row_id, row in enumerate(reader):

                if row_id == 3 or row_id == 5 or row_id == 6:
                    header_rows.append(row)

                elif row_id > 6:
                    rows.append(row)
                else:
                    continue

        headers = ['_'.join([x[i] for x in header_rows if len(x[i]) > 0]) for i in range(len(header_rows[0]))]
        headers = [h.replace(" ", "_").replace("(", "").replace(")", "") for h in headers]

        first_noncamera_column = min([i for (i, h) in enumerate(headers) if ("Marker" in h and "camera" not in h)])

        headers = headers[:first_noncamera_column]
        rows = [row[:first_noncamera_column] for row in rows]

        #FOR DUPLICATING FRAME ROW
        headers = [headers[0]] + headers
        rows = [[row[0]] + row for row in rows]

        with open(cleaned_pose_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
            
