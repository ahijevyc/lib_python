# Copied from http://code.activestate.com/recipes/491261-caching-and-throttling-for-urllib2/ May 16, 2017
import pdb
import http.client
import logging
import hashlib
import urllib.request, urllib.error, urllib.parse
import os
import io
__version__ = (0,1)
__author__ = "Staffan Malmgren <staffan@tomtebo.org>"


class CacheHandler(urllib.request.BaseHandler):
    """Stores responses in a persistant on-disk cache.

    If a subsequent GET request is made for the same URL, the stored
    response is returned, saving time, resources and bandwith"""
    def __init__(self,cacheLocation):
        """The location of the cache directory"""
        self.cacheLocation = cacheLocation
        if not os.path.exists(self.cacheLocation):
            os.mkdir(self.cacheLocation)
            
    def default_open(self,request):
        url = request.full_url
        # Tried passing data to urlopen using optional data argument but 
        # got HTTP Error 403: Forbidden
        if request.data is not None:
            url = url + "?" + request.data.decode()
        if CachedResponse.ExistsInCache(self.cacheLocation, url):
            logging.debug(f"CacheHandler: Returning CACHED response for {url}")
            return CachedResponse(self.cacheLocation, url, setCacheHeader=True)
        else:
            logging.debug(f"CacheHandler: request {url}")
            response = urllib.request.urlopen(url)
            logging.debug(f"CacheHandler: store {url} in {self.cacheLocation}")
            CachedResponse.StoreInCache(self.cacheLocation, url, response)
            return response

class CachedResponse(io.StringIO):
    """An urllib2.response-like object for cached responses.

    To determine wheter a response is cached or coming directly from
    the network, check the x-cache header rather than the object type."""
    
    def ExistsInCache(cacheLocation, url):
        url = url.encode('ascii')
        hash = hashlib.md5(url).hexdigest()
        return (os.path.exists(cacheLocation + "/" + hash + ".headers") and 
                os.path.exists(cacheLocation + "/" + hash + ".body"))
    ExistsInCache = staticmethod(ExistsInCache)

    def StoreInCache(cacheLocation, url, response):
        url = url.encode('ascii')
        hash = hashlib.md5(url).hexdigest()
        with open(cacheLocation + "/" + hash + ".headers", "w") as f:
            headers = str(response.info())
            logging.debug(f"write {hash} headers {headers}")
            f.write(headers)
        with open(cacheLocation + "/" + hash + ".body", "w") as f:
            logging.debug(f"write {hash} body")
            f.write(response.read().decode())
    StoreInCache = staticmethod(StoreInCache)
    
    def __init__(self, cacheLocation,url,setCacheHeader=True):
        self.cacheLocation = cacheLocation
        url = url.encode('ascii')
        hash = hashlib.md5(url).hexdigest()
        io.StringIO.__init__(self, open(self.cacheLocation + "/" + hash+".body").read())
        self.url     = url
        self.code    = 200
        self.msg     = "OK"
        headerbuf = open(self.cacheLocation + "/" + hash+".headers").read()
        if setCacheHeader:
            headerbuf += "d-cache: %s/%s\r\n" % (self.cacheLocation,hash)
        self.headers = http.client.HTTPMessage(io.StringIO(headerbuf))

    def info(self):
        return self.headers
    def geturl(self):
        return self.url

