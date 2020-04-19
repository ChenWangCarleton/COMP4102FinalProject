# COMP4102 Final project

Hi! I'm your first Markdown file in **StackEdit**. If you want to learn about StackEdit, you can read me. If you want to play with Markdown, you can edit me. Once you have finished with me, you can create new files by opening the **file explorer** on the left corner of the navigation bar.
# To run the project
- please ensure python 3.7.3 is installed
- this project is developed and tested on windows 10
## please ensure the version of following python modules
- please ensure numpy version is 1.16.2
- please ensure opencv-contrib-python version is 3.4.2.16
-  please ensure opencv-python version is 3.4.2.16
- please ensure Pillow version is 7.0.0

# Example use case
Put the allinone.py, utils.py and nbaclip_1.mp4 under the same directory
Open a console and change the current working directory to the same directory above
Execute the project by executing :
python .\allinone.py

# Files for this project
- file allinone.py, utils.py and sample data nbaclip_1.mp4 are the essential part of our project deliverable
- files in folder "folder before merging" contains all the python files and screenshots we created during the finishing of this project. Those files are not required to run this project, but are still kept as a proof of our work
- tracker feature explained.pdf explains the tracker that is used in this program that can be used by the user to change some threshold during runtime.
# Summary

Our project provides the user a way to view basketball match videos with special effects applied in real-time. We developed an algorithm to detect momentum shifts within the court by tracking the court borders, players, and the location of the basketball. Overall, our project is able to detect court borders, players, the location of the basketball and apply special effects on certain areas of the video based on analysis in real time.

The project consists of 5 parts, court detection, player detection, ball detection, applying special effects on certain areas, and the algorithm for analyzing momentum shifts on the court.

The project consists of 5 parts, court detection, player detection, ball detection, applying special effects on certain areas, and the algorithm for analyzing momentum shifts on the court. 
1) Court detection – Locating the upper, lower, left, and right borders by using a Canny edge detector and Hough transform. 
2) Player detection – Locating individuals in the videos using histogram of oriented gradients, color block detection and single shot detector. 
3) Ball detection – Detecting circle objects by using blurring and Hough transform. 
4) Applying special effects – Applying convolution with different filter kernels on certain areas of the image. 
5) Analyzing momentum shifts on the court – Analyzing results from detection to mark the correct objects and provide areas for applying special effects. 

The data used for this project is acquired from the NBA official website for game replays. The replays are then edited to remove advertisements and non-game play moments. During the replay, the camera angle is changing, zooming in and out constantly to track the action on the court, which significantly increases the difficulty level for detection. We specifically selected replays that are recorded by one camera only to avoid further increasing the difficulty level of implementing this project.

To detect lines and circles for border detection as well as ball detection, Hough transform is used after applying edge detection methods on the image. Hough transform is a method for isolating features of certain shapes from an image. According to a study in 2013 [1], applying Hough transform after using edge detection method significantly raised the accuracy of line detection. According to another study for tracking basketball players[2], converting an image from BGR to HSV (hue, saturation and value) color model and performing erosion and dilation on the H-plane of the image before using Hough transform will increase the accuracy of detection. We implemented and examined 3 different approaches for player detection. The first approach is detecting contours after filtering images using color threshold. The second approach is using histograms of oriented gradients. The third approach is using a single shot detector. In the end, we used the histograms of oriented gradients method in our final deliverable for player detection. As it has the best accuracy for human detection according to a study in 2005[3].

Equal work was performed by both project members.

Reference

[1] S. Singh1 and A. Datar2, “EDGE Detection Techniques Using Hough Transform.” [Online]. Available: https://pdfs.semanticscholar.org/6dfd/9a799dd18a05b417302a30660fc7302cff52.pdf. [Accessed: 17-Apr-2020]. 
[2] E. Cheshire, C. Halasz, and J. K. Perin, “Player Tracking and Analysis of Basketball Plays.” [Online]. Available: https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Cheshire_Halasz_P erin.pdf. [Accessed: 17-Apr-2020]. 
[3] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 886-893 vol. 1.