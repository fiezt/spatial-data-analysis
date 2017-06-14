import math

class MapOverlay:
    def __init__(self, topleft_latlong, bottomright_latlong, pixels, resolution=1024.0):
        """Initializing the size parameters for the image.

        :param topleft_latlong: List of tuple of the lat, long pair of the 
        top left corner of the image.
        :param bottomright_latlong: List or tuple of the lat, long pair of the 
        bottom right corner of the image.
        :param pixels: List or tuple of the pixel dimensions of the image.
        :param resolution: Float of the projected resolution of the lat, long pair.
        """
        
        self.res = resolution

        self.topleft = self.to_web_mercator(topleft_latlong)
        self.bottomright = self.to_web_mercator(bottomright_latlong)

        # Calculate the scaling factors of the projected coordinates.
        self.horzscale = pixels[0]/(abs(self.bottomright[1] - self.topleft[1]))
        self.vertscale = pixels[1]/(abs(self.topleft[0] - self.bottomright[0]))
        
        
    def to_web_mercator(self, coord, zoomlvl=1):
        """Convert latitude and longitude pair to web mercator pixel position.

        For more information on this see https://en.wikipedia.org/wiki/Web_Mercator.
        
        :param coord: List or tuple of the lat, long pair.
        :param zoomlvl: Scaler zoom level of the image.

        :return pair: Tuple of lat, long pair for pixels.
        """

        lat = coord[0]
        lon = coord[1]
    
        # Latitude conversion.
        lat_rad = lat * math.pi/180.0
        yit = math.pi - math.log(math.tan((0.25*math.pi) + (0.5*lat_rad)))
        y = (self.res)/math.pi * math.pow(2,zoomlvl) * yit
        
        # Longitude conversion.
        lon_rad = lon * math.pi/180.0
        x = (self.res)/math.pi * math.pow(2,zoomlvl) * (lon_rad + math.pi)

        pair = (y, x)
    
        return pair


    def to_image_pixel_position(self, coord):
        """Convert latitude and longitude pair to image pixel position in image.
        
        :param coord: List or tuple of the lat, long pair.

        :return position: Tuple of corresponding pixel position of lat, long pair.
        """

        webmcoord = self.to_web_mercator(coord)

        horz = abs(webmcoord[0] - self.topleft[0])*self.horzscale
        vert = abs(webmcoord[1] - self.topleft[1])*self.vertscale
    
        position = tuple([int(round(vert)), int(round(horz))])
    
        return position