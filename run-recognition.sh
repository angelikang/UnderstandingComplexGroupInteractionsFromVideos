
for VIDEO in 1542 1543 1544 1550 1552 1565 1568 1576 1584 1585 1593 1600 1602 1606
do
    echo 
    echo $VIDEO
    python3m mark-faces.py $VIDEO > ./data/$VIDEO.log
done