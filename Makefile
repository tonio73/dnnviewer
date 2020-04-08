package: package_clean
	python3 setup.py sdist bdist_wheel

package_upload:
	python3 -m twine upload --verbose dist/*

# Login = '__token__'
# Password = token starting with 'pypi-'
package_upload_test:
	python3 -m twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

package_clean:
	rm -f dist/*

.PHONY: package package_upload package_upload_test package_clean