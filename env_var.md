# This script demonstrates how to set up the environment for using the Yandex Cloud API with PowerShell

## It includes steps to install the Yandex Cloud CLI, configure the API key, and set up environment variables

1. Get the folder ID

'''Powershell
yc resource-manager folder list
'''

2. Get the service account ID

'''Powershell
yc iam service-account list
'''

3. Set the environment variables

'''Powershell
set folder_id=your-folder-id
set api_key=your-api-key
'''
4. Test the environment variables

'''Powershell
echo $env:folder_id
echo $env:api_key
'''
