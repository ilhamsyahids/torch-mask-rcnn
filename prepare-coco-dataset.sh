mkdir -p data/coco

STAGE_DIR=coco

echo "Downloading train2017.zip"
wget -O $STAGE_DIR/train2017.zip http://images.cocodataset.org/zips/train2017.zip
echo "Extracting $STAGE_DIR/train2017.zip"
unzip -o $STAGE_DIR/train2017.zip -d $STAGE_DIR | awk 'BEGIN {ORS="="} {if(NR%1000==0)print "="}'
echo

echo "Downloading val2017.zip"
wget -O $STAGE_DIR/val2017.zip http://images.cocodataset.org/zips/val2017.zip
echo "Extracting $STAGE_DIR/val2017.zip"
unzip -o $STAGE_DIR/val2017.zip -d $STAGE_DIR | awk 'BEGIN {ORS="="} {if(NR%1000==0)print "="}'
echo

echo "Downloading annotations_trainval2017.zip"
wget -O $STAGE_DIR/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Extracting $STAGE_DIR/annotations_trainval2017.zip"
unzip -o $STAGE_DIR/annotations_trainval2017.zip -d $STAGE_DIR | awk 'BEGIN {ORS="="} {if(NR%1000==0)print "="}'
