import math

class MapOverlay:
    def __init__(self, topleft_latlong, bottomright_latlong, pixels, resolution=1024.0):
        #resolution is the projected resolution of the latitude and longitude coordinates
        #to integer pixel values--a higher projected resolution results in coordinate resolution
        #per pixel
        
        #topleft_latlong and bottomright_latlong coorespond to the upper right and bottom left
        #latitude and longitude coordinates visible in your Mercator projected map image
        self.res = resolution
        self.topleft = self.to_web_mercator(topleft_latlong)
        self.bottomright = self.to_web_mercator(bottomright_latlong)
        
        #the following returns the vertical and horizontal scaling factor of the projected coordinates to 
        #the pixel size of the map image
        #ex: pixels = [256,256]
        self.horzscale = pixels[0]/(abs(self.bottomright[1] - self.topleft[1]))
        self.vertscale = pixels[1]/(abs(self.topleft[0] - self.bottomright[0]))
        
    def to_web_mercator(self, coord, zoomlvl=1):
        #raw latitude longitude pair to web mercator pixel position
        #https://en.wikipedia.org/wiki/Web_Mercator
        #1024x1024 base pixel image
        #x = longitude
        #y = latitude, all converted coordinate pairs are read as [latitude, longitude]
        lat = coord[0]
        lon = coord[1]
    
        #latitude conversion
        lat_rad = lat * math.pi/180.0
        yit = math.pi - math.log(math.tan( (0.25*math.pi) + (0.5*lat_rad) ))
        y = (self.res)/math.pi * math.pow(2,zoomlvl) * yit
        
        #longitude conversion
        lon_rad = lon * math.pi/180.0
        x = (self.res)/math.pi * math.pow(2,zoomlvl) * (lon_rad + math.pi)
    
        return([y,x])

    def to_image_pixel_position(self, coord):
        #raw latitude longitude pair to image pixel position
        #lat --> vertical scale
        #long --> horizontal scale  
        webmcoord = self.to_web_mercator(coord)
        horz = abs(webmcoord[0] - self.topleft[0])*self.horzscale
        vert = abs(webmcoord[1] - self.topleft[1])*self.vertscale
    
        position = [int(round(vert)), int(round(horz))]
    
        return(position)