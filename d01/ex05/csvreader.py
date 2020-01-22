import csv

class CsvReader():

	def __init__(self, filename, sep=',', header=False, skip_top=0, skip_bottom=0):
	    self.filename = filename
	    self.sep = sep
	    self.has_header = header
	    self.skip_top = skip_top
	    self.skip_bottom = skip_bottom
	    self.file = None
	    self.data = None
	    self.header = None

	def __enter__(self):
	    try:
	        self.file = open(self.filename, "r")
	    except:
	        print("file not found")
	        return(self)
	    print("Opened")
	    
	    reader = csv.reader(self.file, delimiter = self.sep)
	    for truc in range(self.skip_top):
	        next(reader)
	    if self.has_header:
	        self.header = next(reader)
	    self.data = [line for line in reader]
	    if self.skip_bottom:
	        self.data = self.data[:-self.skip_bottom]
	    #check if file is corrupted --> does not work
	    num = len(self.data[0])
	    #print(num)
	    for line in self.data:
	        if len(line) != num:
	            return(None)
	    return(self)

	def __exit__(self, exc_type, exc_value, exc_traceback):
	    if self.data != None:
	        print("Closed")
	        self.file.close()

	def getdata(self):
            return(self.data)

	def getheader(self):
	    return(self.header)

if __name__ == "__main__":
	with CsvReader('bad.csv', header = True, skip_top = 0, skip_bottom = 1) as csv_file:
		data = csv_file.getdata()
		header = csv_file.getheader()
		print("header: \n" + str(header) + "\n")
		if data:
			for lines in data:
				print(lines)
