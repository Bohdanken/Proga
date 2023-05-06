# AIR RAID ALARM FORECAST API AND​  THE CORRESPONDING UI

We decided to create a service that will predict air raid alarms by hour for every region in Ukraine, based on the weather data and the reports from the Institute for the Study of War (ISW). We believe this will help users plan their days accordingly and prepare them for possible danger. Also, our Air Raid Alarm Forecast API (ARAF API) can help other developers create something important.  We  haven‘t  found  any  similar  products in the  Web, so we  are  proud  to  say, that  we  are  now  filling  this  gap.​



## Installation

To run this project on your machine, follow these steps:

1.  Clone the repository to your local machine: `git clone https://github.com/Bohdanken/Proga`
2. Clone branch ServerBranch on your server
    
3.  Install the required dependencies by running the following command in the project directory: `pip install -r requirements.txt`
    
4.  Download the necessary data files and place them in the appropriate directories in the project folder.
    
5.  Run the project by executing the following command:  `pyuwsgi --http 0.0.0.0:7000 --wsgi-file app.py --callable app --processes 4 --threads 2 --stats 127.0.0.1:9191`
    

## Usage of API

Once the project is running, it will continuously take the weather archive and make new predictions when an alarm will be every hour.  These predictions can be accessed through the project's API, which can be accessed by making HTTP requests to the following endpoint:

`http://your_server's_ip:port/send_prediction`

The endpoint returns a JSON object containing the predicted air alarm times for the current day. The predictions are updated hourly, so it is recommended to make a new request every hour.

## Usage of Frontend

You should imply that  site uses our API.

 Enter localhost address of your api on the server in variable of `app1.py` . Then start this file the same way as API. Now it runs in parallel with API. You can access it online by entering ip and port in browser.
    
 The frontend displays the predicted air alarm times for the current day as obtained from the API endpoint `http://your_server's_ip:port/send_prediction`. The predicted times are updated hourly, so you can take almost any prediction.
    

## Contributing

Contributions to this project are welcome. To contribute, follow these steps:

1.  Fork the repository.
    
2.  Create a new branch for your changes: `git checkout -b my-feature-branch`
    
3.  Make your changes and commit them: `git commit -am "Add new feature"`
    
4.  Push your changes to your fork: `git push origin my-feature-branch`
    
5.  Submit a pull request to the upstream repository.
    

## License

This project is not licensed.

## Contact
shevchenkobogdan2004@gmail.com
