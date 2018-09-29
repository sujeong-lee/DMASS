def radec2le(ra,dec):
	from math import pi,cos,sin,atan2,asin
	deg2Rad = pi/180.0
	surveyCenterDEC = 32.5
	surveyCenterRA = 185.0
	etaPole = deg2Rad*surveyCenterDEC
	node = deg2Rad*(surveyCenterRA - 90.0)
	
	x = cos(deg2Rad*ra-node)*cos(deg2Rad*dec)
  	y = sin(deg2Rad*ra-node)*cos(deg2Rad*dec)
  	z = sin(deg2Rad*dec)

  	lam = -1.0*asin(x)/deg2Rad
  	eta = (atan2(z,y) - etaPole)/deg2Rad
  	if eta < -180.0:
  		eta += 360.0
  	if eta > 180.0:
  		eta -= 360.0
  	
  	return (lam,eta)


def ang2stripe(ra,dec):
	eta = radec2le(ra,dec)[1]
	inc = eta + 32.5
	stripe = int(inc/2.5 + 10)
	return stripe
