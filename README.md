# predictive_maintainance
**This is a project for the course Big Data Analysis and Business Intelligence**
<br />
<br />
To run this project, first download NASA Bearing Dataset from Kaggle and put it in the same directory with this repository. For example:<br />
<img width="231" height="275" alt="{A0DE8D96-EB21-4352-AABC-979B5A3C5F41}" src="https://github.com/user-attachments/assets/991ddfd3-f21c-4aca-9553-f39b1425d750" />
<br />
After that change the directory mapping in docker-compose.yml file to:
  producer:
    build: ./Producer
    depends_on:
      - kafka
    volumes:
      # Map the data folder from your laptop into the container
      - ./<your_path>:/app/NASA_Bearing_Data

Finally run docker-compose up --build to run the project
and run docker-compose down -v to stop.
<br />
<br />
<br />
**Please remember to change from CRLF to LF in ./Processing/start.sh.**
<br />

**In VS Code:**
<br />
1. Open Processing/start.sh.
<br />
2. Look at the bottom right corner of the window. It probably says CRLF.
<br />
3. Click CRLF and change it to LF.
<br />
4. Save the file.
