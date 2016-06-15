# Installs ICA-AROMA to a centos image with FSL pre-installed

# function provided by Tristan A.A., ttaa9 on github

FROM mcin/docker-fsl:latest

# Install necessary python packages
RUN yum update -y; yum clean all
RUN yum install -y numpy scipy

# Add everything to the container
ADD . /ICA-AROMA

