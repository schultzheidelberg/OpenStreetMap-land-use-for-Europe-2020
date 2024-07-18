# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:41:03 2019

@author: Janek
"""
# import necessary libraries
# gdal and psycopg2 libraries need to be installed for this script to work
import sys
import os, glob
import ogr,osr
import psycopg2 as pg
import time, datetime
# for logging progress
from functools import wraps
import logging
import traceback

def log(f):
    @wraps(f)
    # decoration function which will enhance following function with some info about them
    def wrapper(*args, **kwds):

        # set logger level
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # configure handler (console)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)
        
        # remove previous handlers
        if (logger.hasHandlers()):
            logger.handlers.clear()
            
        # add handler    
        logger.addHandler(consoleHandler)
        
        # set logging format
        formatter = logging.Formatter("%(asctime)-15s %(levelname)s - %(message)s")
        consoleHandler.setFormatter(formatter)
                
        start, stop = 0, 0
        ret = False

        
        if logger:
            start = time.time()
            logger.info("Enterning {method}()".format(method=f.__name__))

        try:
            ret = f(*args, **kwds)

        except Exception:
            if logger:
                error = traceback.format_exc()
                for line in error.split("\n"):
                    logger.error(line)

        if logger:
            stop = time.time()
            logger.info(
                "Leaving {method}(), elapsed time {elapsed_time}".format(
                    method=f.__name__,
                    elapsed_time=datetime.timedelta(
                        milliseconds=(stop-start)*1000)
                )
            )
        return ret

    return wrapper

class config(object):
   def __init__(self): 
       self.root_dir = os.path.dirname(__file__)
       self.shp_dir  = input("Please specify the location/folder of the Shapefile(s): ")

       if os.path.exists(self.shp_dir):
          os.chdir(str(self.shp_dir))
          print ("Sucessfully changed working directory to {f}".format(f=self.shp_dir))     
       else:
          print ("Could not navigate to {f}. Script could not be executed.".format(f=self.shp_dir))
          sys.exit(0)   

       if len(glob.glob("*.shp")) == 0:
          print ("No Shapefile(s) found in {folder}. Script could not be executed.".format(folder=os.getcwd())) 
          sys.exit()

       self.bboxes = []
       self.filenames = []
       self.db_con = None
       self.cursor = None
      
       for f in (self.get_extent, self.pg_connect, self.sql_query):
          ret = f()
          if not ret:
             print("A module of the script could not be executed.")
             sys.exit(0)
             
       print ("Resulting LULC-shp file(s) saved under:{outpath}".format(outpath=self.root_dir + '/results/'))

   @log 
   def get_extent(self):
         ret = False  
         try:
            driver = ogr.GetDriverByName('ESRI Shapefile')
            # get all shapefiles in folder       
            for shp_file in glob.glob("*.shp"):
               index = 0
               filename = "feature_"
               shapefile = driver.Open(shp_file)
               layer = shapefile.GetLayer()

               # check coordinate system
               srs = layer.GetSpatialRef()      
               if srs.GetAttrValue("AUTHORITY", 1) != '4326':
                  print('Projection of the Shapefile {shp} should be WGS84.'.format(shp=shp_file))
                  return

               # get all features of each shapefile in folder
               feature_counter = layer.GetFeatureCount()
               for i in range(feature_counter):
                  index += 1
                  feature = layer.GetFeature(i)
                  # get the extent of the first feature
                  extent = feature.GetGeometryRef().GetEnvelope()
                  minX, maxX, minY, maxY = extent
                  # get area of each feature
                  area = feature.GetGeometryRef().GetArea()
                  # bounding boxes are split into smaller one for bigger areas (bigger than about 1000km² - adjust this values if errors occur)
                  if area > 50:
                     new_bboxes = split_bbox(minX ,minY ,maxX ,maxY)
                     for i in range (len(new_bboxes)):
                        self.bboxes.append(new_bboxes[i])
                        self.filenames.append(filename + str(index) + '_extract_' + str(i+1))
                  # collect extents of smaller bboxes
                  else:
                     self.bboxes.append((minX, minY, maxX, maxY))
                     self.filenames.append(filename + str(index))

               shapefile = None

            ret = True

         except:
            print("Could not retrieve the bbox from the shapefile.")
 
         return ret

   @log
   def pg_connect(self):
        ret = False     
        try:
           self.db_con = pg.connect(dbname="osm_hstore", user="osmuserread", password="ASKFROMMICHEL", host="osmosis.geog.uni-heidelberg.de", port="5432")
           self.cursor = self.db_con.cursor()
           ret = True
        
        except:
            print("Could not connect to database. Make sure the parameters are correct")
        
        return ret

   @log  
   def sql_query(self):
      ret = False
      try:
         
         # store results in folder
         if not os.path.exists(self.root_dir + '/results/'):
                os.makedirs(self.root_dir + '/results/')
         os.chdir(self.root_dir + '/results/')
         
         index = 0
         # for each bbox a query is executed
         for box in self.bboxes:
            
     
            query = '''
            SELECT
     	CASE
     		WHEN 
     			nat.the_geom && st_transform(st_makeenvelope({x},{y},{xx},{yy}, 4326),3857)-- a && b : a and b's bbox intersect
     			AND
     			nat.the_geom @  st_transform(st_makeenvelope({x},{y},{xx},{yy}, 4326),3857) -- a @ b : a completly contained by b
     		THEN ST_AsText(nat.the_geom)
     	END as the_geom,
        CASE 
			WHEN type IN ('garages', 'residential') THEN 'urban_fabric'
			WHEN type IN ('railway', 'industrial', 'commercial', 'retail', 'harbour', 'port', 'lock', 'marina') THEN 'indu_comm'
			WHEN type IN ('quarry', 'construction', 'landfill', 'brownfield') THEN 'mine_dump'
			WHEN type IN ('stadium', 'recreation_ground', 'golf_course', 'sports_center', 'playground', 'pitch', 'village_green', 'allotments', 'cemetery', 'park', 'zoo', 'track', 'garden', 'raceway') THEN 'artificial_veg'
			WHEN type IN ('greenhouse_horticulture', 'greenhouse', 'farmland', 'farm', 'farmyard') THEN 'arable_land'
            WHEN type IN ('vineyard', 'orchard') THEN 'permanent_crops'
			WHEN type = 'meadow' THEN 'pastures'
			WHEN type IN ('forest', 'wood') THEN 'forests'
			WHEN type IN ('grass', 'greenfield', 'scrub', 'heath', 'grassland') THEN 'shrub'
			WHEN type IN ('cliff', 'fell', 'sand', 'scree', 'beach', 'mud', 'glacier', 'rock') THEN 'open_spaces'
			WHEN type IN ('marsh', 'wetland') THEN 'inland_wetlands'
			WHEN type IN ('salt_pond', 'tidal') THEN 'costal_wetland'
			WHEN type IN ('water', 'riverbank', 'reservoir', 'basin', 'dock', 'canal', 'pond') THEN 'water_bodies'
        END as class
  FROM public.naturals as nat
 WHERE ST_Intersects(nat.the_geom, st_transform(st_makeenvelope({x},{y},{xx},{yy}, 4326),3857)) AND type in ('garages', 'residential','railway', 'industrial', 'commercial', 'retail', 'harbour', 'port', 'lock', 'marina','quarry', 'construction', 'landfill', 'brownfield','stadium', 'recreation_ground', 'golf_course', 'sports_center', 'playground', 'pitch', 'village_green', 'allotments', 'cemetery', 'park', 'zoo', 'track', 'garden', 'raceway','greenhouse_horticulture', 'greenhouse', 'farmland', 'farm', 'farmyard','vineyard', 'orchard','meadow','forest', 'wood','grass', 'greenfield', 'scrub', 'heath', 'grassland','cliff', 'fell', 'sand', 'scree', 'beach', 'mud', 'glacier', 'rock','marsh', 'wetland','salt_pond', 'tidal','water', 'riverbank', 'reservoir', 'basin', 'dock', 'canal', 'pond')
'''.format(x = box[0], y = box[1], xx= box[2], yy = box[3])

            self.cursor.execute(query)
            obj = self.cursor.fetchall()
            file = self.filenames[index]
            
            # execute another function for the export
            self.export_to_shape(obj,file)     
            index += 1      
            print (file + " done")            
                
         # close connection when done
         self.cursor.close() 
         self.db_con.close()
         os.chdir(self.root_dir)
         
         ret = True
      except:
         print ("Query is too large to fetch locally. Try using server sides cursors or reduce the bbox")
      return ret
   
   @log
   def export_to_shape(self, query_object, filename):
      try:
         # Create the output Driver
         outDriver = ogr.GetDriverByName('ESRI Shapefile')
   
         # define projection
         proj = osr.SpatialReference()
         proj.ImportFromEPSG(3857)
   
         # create shp file
         outputshp = "{path}/results/{file}.shp".format(path=self.root_dir, file=filename)
         

         if os.path.exists(outputshp):
            outDriver.DeleteDataSource(outputshp)
         outDataSet = outDriver.CreateDataSource(outputshp)
         outLayer = outDataSet.CreateLayer(filename, proj, ogr.wkbPolygon)
         
         # create field
         field_definition = ogr.FieldDefn("class", ogr.OFTString)
         field_definition.SetWidth(24)
         outLayer.CreateField(field_definition)
         
         # create second field
         field_definition2 = ogr.FieldDefn("code", ogr.OFTString)
         field_definition2.SetWidth(4)
         outLayer.CreateField(field_definition2)         
         
         # iterate through lines of the query 
         for line in query_object:
            if line[0] == None: # if empty geom -> pass this line
               continue
            # create the feature
            feature = ogr.Feature(outLayer.GetLayerDefn())
            # set the field values
            feature.SetField("class", line[1])
            
            # add class code to second field
            if line[1] == "urban_fabric":
               feature.SetField("code","1.1")
            if line[1] == "indu_comm":
               feature.SetField("code","1.2")
            if line[1] == "mine_dump":
               feature.SetField("code","1.3")
            if line[1] == "artificial_veg":
               feature.SetField("code","1.4")
            if line[1] == "arable_land":
               feature.SetField("code","2.1")
            if line[1] == "permanent_crops":
               feature.SetField("code","2.2")
            if line[1] == "pastures":
               feature.SetField("code","2.3")
            if line[1] == "forests":
               feature.SetField("code","3.1")
            if line[1] == "shrub":
               feature.SetField("code","3.2")
            if line[1] == "open_spaces":
               feature.SetField("code","3.3")
            if line[1] == "inland_wetlands":
               feature.SetField("code","4.1")
            if line[1] == "costal_wetland":
               feature.SetField("code","4.2")
            if line[1] == "water_bodies":
               feature.SetField("code","5.0")

            # add geometry
            feature.SetGeometry(ogr.CreateGeometryFromWkt(line[0]))        
            outLayer.CreateFeature(feature)
            # close everything
            del feature       
         del outDataSet         
      except:
         print ("Could not write Shapefile of file {file}".format(file=filename))

def split_bbox(minX, minY, maxX, maxY):
   # the initial extent/bbox of a feature is split into smaller bounding boxes with the size of x degrees (WGS84).
   # The shift is performed n times, depending on its width and height
   # width and height, repectively divided by x and converted to integer value results in n. 
   # +1 is added to width and height to ensure to include the complete bbox (otherwise a gap would be left over) 
   bboxes = []
   size = int(4) # size of extracts in both x and y dimension (in degrees)
   
   numberx = int(((int(maxX)-int(minX))/size)) + 1
   numbery = int(((int(maxY)-int(minY))/size)) + 1
   extracts = numberx * numbery
   print ("The BBox of a feature inside a shapefile is too big to process as a whole. It is now split into {extracts} extracts/files.".format(extracts=extracts))
   print ("Please wait...")

   minx_2 = minX # the initial minimum x value is set to the minimum x values of the complete feature
   for x in range(numberx):
      miny_2 = minY # initial minimum y value is set to the minimum y value of the complete feature
      maxx_2 = minx_2 + size 
      for y in range(numbery):
         maxy_2 = miny_2 + size
         if maxx_2 > maxX: # if a width value exceeds the initial bbox maximum x-Value, it is trimmed to x-max (this happens in the last run of the loop due to the +1 in n+1)
            maxx_2 = maxX
         if maxy_2 > maxY: # the height value is treated in the same fashion
            maxy_2 = maxY
         bboxes.append((minx_2,miny_2,maxx_2,maxy_2))
         miny_2 += size
      minx_2 += size

   return bboxes

if __name__ == "__main__":
   starttime = time.time()
   config() 
   endtime = time.time()
   print("Elapsed time {elapsed_time}".format(elapsed_time=datetime.timedelta(milliseconds=(endtime-starttime)*1000)))     