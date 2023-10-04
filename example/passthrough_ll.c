/*
  FUSE: Filesystem in Userspace
  Copyright (C) 2001-2007  Miklos Szeredi <miklos@szeredi.hu>

  This program can be distributed under the terms of the GNU GPLv2.
  See the file COPYING.
*/

/** @file
 *
 * This file system mirrors the existing file system hierarchy of the
 * system, starting at the root file system. This is implemented by
 * just "passing through" all requests to the corresponding user-space
 * libc functions. In contrast to passthrough.c and passthrough_fh.c,
 * this implementation uses the low-level API. Its performance should
 * be the least bad among the three, but many operations are not
 * implemented. In particular, it is not possible to remove files (or
 * directories) because the code necessary to defer actual removal
 * until the file is not opened anymore would make the example much
 * more complicated.
 *
 * When writeback caching is enabled (-o writeback mount option), it
 * is only possible to write to files for which the mounting user has
 * read permissions. This is because the writeback cache requires the
 * kernel to be able to issue read requests for all files (which the
 * passthrough filesystem cannot satisfy if it can't read the file in
 * the underlying filesystem).
 *
 * Compile with:
 *
 *     gcc -Wall passthrough_ll.c `pkg-config fuse3 --cflags --libs` -o passthrough_ll
 *
 * ## Source code ##
 * \include passthrough_ll.c
 */

#define _GNU_SOURCE
#define FUSE_USE_VERSION 34

#include <fuse_lowlevel.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <dirent.h>
#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/xattr.h>

#include "passthrough_helpers.h"

/* We are re-using pointers to our `struct lo_inode` and `struct
   lo_dirp` elements as inodes. This means that we must be able to
   store uintptr_t values in a fuse_ino_t variable. The following
   incantation checks this condition at compile time. */
#if defined(__GNUC__) && (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 6) && !defined __cplusplus
_Static_assert(sizeof(fuse_ino_t) >= sizeof(uintptr_t),
	       "fuse_ino_t too small to hold uintptr_t values!");
#else
struct _uintptr_to_must_hold_fuse_ino_t_dummy_struct \
	{ unsigned _uintptr_to_must_hold_fuse_ino_t:
			((sizeof(fuse_ino_t) >= sizeof(uintptr_t)) ? 1 : -1); };
#endif

// David: exists for each file/dir after do_lookup. Pointer will be given as a parameter to each method
struct lo_inode {
	struct lo_inode *next; /* protected by lo->mutex */
	struct lo_inode *prev; /* protected by lo->mutex */
	int fd;
	ino_t ino;
	dev_t dev;
	uint64_t refcount; /* protected by lo->mutex */
};

enum {
	CACHE_NEVER,
	CACHE_NORMAL,
	CACHE_ALWAYS,
};

// David: only 1 lo_data instance exists, holding the pointer to root
struct lo_data {
	pthread_mutex_t mutex;
	int debug;
	int writeback;
	int flock;
	int xattr;
	char *source;
	double timeout;
	int cache;
	int timeout_set;
	struct lo_inode root; /* protected by lo->mutex */
};

static const struct fuse_opt lo_opts[] = {
	{ "writeback",
	  offsetof(struct lo_data, writeback), 1 },
	{ "no_writeback",
	  offsetof(struct lo_data, writeback), 0 },
	{ "source=%s",
	  offsetof(struct lo_data, source), 0 },
	{ "flock",
	  offsetof(struct lo_data, flock), 1 },
	{ "no_flock",
	  offsetof(struct lo_data, flock), 0 },
	{ "xattr",
	  offsetof(struct lo_data, xattr), 1 },
	{ "no_xattr",
	  offsetof(struct lo_data, xattr), 0 },
	{ "timeout=%lf",
	  offsetof(struct lo_data, timeout), 0 },
	{ "timeout=",
	  offsetof(struct lo_data, timeout_set), 1 },
	{ "cache=never",
	  offsetof(struct lo_data, cache), CACHE_NEVER },
	{ "cache=auto",
	  offsetof(struct lo_data, cache), CACHE_NORMAL },
	{ "cache=always",
	  offsetof(struct lo_data, cache), CACHE_ALWAYS },

	FUSE_OPT_END
};

static void passthrough_ll_help(void)
{
	printf(
"    -o writeback           Enable writeback\n"
"    -o no_writeback        Disable write back\n"
"    -o source=/home/dir    Source directory to be mounted\n"
"    -o flock               Enable flock\n"
"    -o no_flock            Disable flock\n"
"    -o xattr               Enable xattr\n"
"    -o no_xattr            Disable xattr\n"
"    -o timeout=1.0         Caching timeout\n"
"    -o timeout=0/1         Timeout is set\n"
"    -o cache=never         Disable cache\n"
"    -o cache=auto          Auto enable cache\n"
"    -o cache=always        Cache always\n");
}

// David: get info about the file system config or root inode
static struct lo_data *lo_data(fuse_req_t req)
{
	return (struct lo_data *) fuse_req_userdata(req);
}

// David: either get the root inode or convert FUSE's inode pointer to the custom inode struct
static struct lo_inode *lo_inode(fuse_req_t req, fuse_ino_t ino)
{
	if (ino == FUSE_ROOT_ID)
		return &lo_data(req)->root;
	else
		return (struct lo_inode *) (uintptr_t) ino;
}

// David: get the fd for the inode
static int lo_fd(fuse_req_t req, fuse_ino_t ino)
{
	return lo_inode(req, ino)->fd;
}

static bool lo_debug(fuse_req_t req)
{
	return lo_data(req)->debug != 0;
}

// David: retrieve options provided in fuse_session_new (see main)
static void lo_init(void *userdata,
		    struct fuse_conn_info *conn)
{
	struct lo_data *lo = (struct lo_data*) userdata;

	if(conn->capable & FUSE_CAP_EXPORT_SUPPORT)
		conn->want |= FUSE_CAP_EXPORT_SUPPORT;

	if (lo->writeback &&
	    conn->capable & FUSE_CAP_WRITEBACK_CACHE) {
		if (lo->debug)
			fuse_log(FUSE_LOG_DEBUG, "lo_init: activating writeback\n");
		conn->want |= FUSE_CAP_WRITEBACK_CACHE;
	}
	if (lo->flock && conn->capable & FUSE_CAP_FLOCK_LOCKS) {
		if (lo->debug)
			fuse_log(FUSE_LOG_DEBUG, "lo_init: activating flock locks\n");
		conn->want |= FUSE_CAP_FLOCK_LOCKS;
	}
}

// David: free all custom inode structs (not sure why userdata doesn't need to be freed)
static void lo_destroy(void *userdata)
{
	struct lo_data *lo = (struct lo_data*) userdata;

	while (lo->root.next != &lo->root) {
		struct lo_inode* next = lo->root.next;
		lo->root.next = next->next;
		free(next);
	}
}

// David: Type = read
static void lo_getattr(fuse_req_t req, fuse_ino_t ino,
			     struct fuse_file_info *fi)
{
	int res;
	struct stat buf;
	struct lo_data *lo = lo_data(req);

	(void) fi;

	res = fstatat(lo_fd(req, ino), "", &buf, AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW);
	if (res == -1)
		return (void) fuse_reply_err(req, errno);

	fuse_reply_attr(req, &buf, lo->timeout);
}

// David: Type = write
static void lo_setattr(fuse_req_t req, fuse_ino_t ino, struct stat *attr,
		       int valid, struct fuse_file_info *fi)
{
	int saverr;
	char procname[64];
	struct lo_inode *inode = lo_inode(req, ino);
	int ifd = inode->fd;
	int res;

	// David: Change permissions
	if (valid & FUSE_SET_ATTR_MODE) {
		if (fi) {
			res = fchmod(fi->fh, attr->st_mode);
		} else {
			// David: If the file has not been opened by the client, assume it has been opened by fuse and get the fd. Not sure why it doesn't just use fchmod(ifd, ...) though.
			sprintf(procname, "/proc/self/fd/%i", ifd);
			res = chmod(procname, attr->st_mode);
		}
		if (res == -1)
			goto out_err;
	}
	// David: Set uid and gid
	if (valid & (FUSE_SET_ATTR_UID | FUSE_SET_ATTR_GID)) {
		uid_t uid = (valid & FUSE_SET_ATTR_UID) ?
			attr->st_uid : (uid_t) -1;
		gid_t gid = (valid & FUSE_SET_ATTR_GID) ?
			attr->st_gid : (gid_t) -1;

		res = fchownat(ifd, "", uid, gid,
			       AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW);
		if (res == -1)
			goto out_err;
	}
	// David: Truncate
	if (valid & FUSE_SET_ATTR_SIZE) {
		if (fi) {
			res = ftruncate(fi->fh, attr->st_size);
		} else {
			sprintf(procname, "/proc/self/fd/%i", ifd);
			res = truncate(procname, attr->st_size);
		}
		if (res == -1)
			goto out_err;
	}
	// David: Change last accessed and modified times
	if (valid & (FUSE_SET_ATTR_ATIME | FUSE_SET_ATTR_MTIME)) {
		struct timespec tv[2];

		tv[0].tv_sec = 0;
		tv[1].tv_sec = 0;
		tv[0].tv_nsec = UTIME_OMIT;
		tv[1].tv_nsec = UTIME_OMIT;

		if (valid & FUSE_SET_ATTR_ATIME_NOW)
			tv[0].tv_nsec = UTIME_NOW;
		else if (valid & FUSE_SET_ATTR_ATIME)
			tv[0] = attr->st_atim;

		if (valid & FUSE_SET_ATTR_MTIME_NOW)
			tv[1].tv_nsec = UTIME_NOW;
		else if (valid & FUSE_SET_ATTR_MTIME)
			tv[1] = attr->st_mtim;

		if (fi)
			res = futimens(fi->fh, tv);
		else {
			sprintf(procname, "/proc/self/fd/%i", ifd);
			res = utimensat(AT_FDCWD, procname, tv, 0);
		}
		if (res == -1)
			goto out_err;
	}

	return lo_getattr(req, ino, fi);

out_err:
	saverr = errno;
	fuse_reply_err(req, saverr);
}

// David: Find the custom inode struct for the given inode number and increments the reference count. Linearly searches a linked list, but should be fine(?) because new inodes are created at the front of the list (after root)
static struct lo_inode *lo_find(struct lo_data *lo, struct stat *st)
{
	struct lo_inode *p;
	struct lo_inode *ret = NULL;

	pthread_mutex_lock(&lo->mutex);
	for (p = lo->root.next; p != &lo->root; p = p->next) {
		if (p->ino == st->st_ino && p->dev == st->st_dev) {
			assert(p->refcount > 0);
			ret = p;
			ret->refcount++;
			break;
		}
	}
	pthread_mutex_unlock(&lo->mutex);
	return ret;
}

// David: Open the file with the given name, finds and returns its custom inode. A new custom inode is created if one does not exist.
// David: Type = write (opens file, creates inode, increments refcount)
static int lo_do_lookup(fuse_req_t req, fuse_ino_t parent, const char *name,
			 struct fuse_entry_param *e)
{
	int newfd;
	int res;
	int saverr;
	struct lo_data *lo = lo_data(req);
	struct lo_inode *inode;

	memset(e, 0, sizeof(*e));
	e->attr_timeout = lo->timeout;
	e->entry_timeout = lo->timeout;

	// David: Open the file, get its fd
	newfd = openat(lo_fd(req, parent), name, O_PATH | O_NOFOLLOW);
	if (newfd == -1)
		goto out_err;

	// David: Check if we can get info about the file? Not sure why this is needed
	res = fstatat(newfd, "", &e->attr, AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW);
	if (res == -1)
		goto out_err;

	// David: Either retrieve its custom inode or create a new one and add it to the root's linked list
	inode = lo_find(lo_data(req), &e->attr);
	if (inode) {
		close(newfd);
		newfd = -1;
	} else {
		struct lo_inode *prev, *next;

		saverr = ENOMEM;
		inode = calloc(1, sizeof(struct lo_inode));
		if (!inode)
			goto out_err;

		inode->refcount = 1;
		inode->fd = newfd;
		inode->ino = e->attr.st_ino;
		inode->dev = e->attr.st_dev;

		pthread_mutex_lock(&lo->mutex);
		prev = &lo->root;
		next = prev->next;
		next->prev = inode;
		inode->next = next;
		inode->prev = prev;
		prev->next = inode;
		pthread_mutex_unlock(&lo->mutex);
	}
	e->ino = (uintptr_t) inode;

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "  %lli/%s -> %lli\n",
			(unsigned long long) parent, name, (unsigned long long) e->ino);

	return 0;

out_err:
	saverr = errno;
	if (newfd != -1)
		close(newfd);
	return saverr;
}

// Type: write (uses lo_do_lookup)
static void lo_lookup(fuse_req_t req, fuse_ino_t parent, const char *name)
{
	struct fuse_entry_param e;
	int err;

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "lo_lookup(parent=%" PRIu64 ", name=%s)\n",
			parent, name);

	err = lo_do_lookup(req, parent, name, &e);
	if (err)
		fuse_reply_err(req, err);
	else
		fuse_reply_entry(req, &e);
}

// Type: write (uses lo_do_lookup)
static void lo_mknod_symlink(fuse_req_t req, fuse_ino_t parent,
			     const char *name, mode_t mode, dev_t rdev,
			     const char *link)
{
	int res;
	int saverr;
	struct lo_inode *dir = lo_inode(req, parent);
	struct fuse_entry_param e;

	// David: Create the node
	res = mknod_wrapper(dir->fd, name, link, mode, rdev);

	saverr = errno;
	if (res == -1)
		goto out;

	// David: Open it (get the fd), create its custom inode
	saverr = lo_do_lookup(req, parent, name, &e);
	if (saverr)
		goto out;

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "  %lli/%s -> %lli\n",
			(unsigned long long) parent, name, (unsigned long long) e.ino);

	fuse_reply_entry(req, &e);
	return;

out:
	fuse_reply_err(req, saverr);
}

// Type: write (uses lo_mknod_symlink)
static void lo_mknod(fuse_req_t req, fuse_ino_t parent,
		     const char *name, mode_t mode, dev_t rdev)
{
	lo_mknod_symlink(req, parent, name, mode, rdev, NULL);
}

// Type: write (uses lo_mknod_symlink)
static void lo_mkdir(fuse_req_t req, fuse_ino_t parent, const char *name,
		     mode_t mode)
{
	lo_mknod_symlink(req, parent, name, S_IFDIR | mode, 0, NULL);
}

// Type: write (uses lo_mknod_symlink)
static void lo_symlink(fuse_req_t req, const char *link,
		       fuse_ino_t parent, const char *name)
{
	lo_mknod_symlink(req, parent, name, S_IFLNK, 0, link);
}

// Type: write (increments inode refcount)
static void lo_link(fuse_req_t req, fuse_ino_t ino, fuse_ino_t parent,
		    const char *name)
{
	int res;
	struct lo_data *lo = lo_data(req);
	struct lo_inode *inode = lo_inode(req, ino);
	struct fuse_entry_param e;
	char procname[64];
	int saverr;

	memset(&e, 0, sizeof(struct fuse_entry_param));
	e.attr_timeout = lo->timeout;
	e.entry_timeout = lo->timeout;

	// David: link
	sprintf(procname, "/proc/self/fd/%i", inode->fd);
	res = linkat(AT_FDCWD, procname, lo_fd(req, parent), name,
		     AT_SYMLINK_FOLLOW);
	if (res == -1)
		goto out_err;

	// David: Check if we can still open the file
	res = fstatat(inode->fd, "", &e.attr, AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW);
	if (res == -1)
		goto out_err;

	// David: Increase inode reference count, then point to this inode
	pthread_mutex_lock(&lo->mutex);
	inode->refcount++;
	pthread_mutex_unlock(&lo->mutex);
	e.ino = (uintptr_t) inode;

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "  %lli/%s -> %lli\n",
			(unsigned long long) parent, name,
			(unsigned long long) e.ino);

	fuse_reply_entry(req, &e);
	return;

out_err:
	saverr = errno;
	fuse_reply_err(req, saverr);
}

// Type: write
static void lo_rmdir(fuse_req_t req, fuse_ino_t parent, const char *name)
{
	int res;

	res = unlinkat(lo_fd(req, parent), name, AT_REMOVEDIR);

	fuse_reply_err(req, res == -1 ? errno : 0);
}

// Type: write
static void lo_rename(fuse_req_t req, fuse_ino_t parent, const char *name,
		      fuse_ino_t newparent, const char *newname,
		      unsigned int flags)
{
	int res;

	if (flags) {
		fuse_reply_err(req, EINVAL);
		return;
	}

	res = renameat(lo_fd(req, parent), name,
			lo_fd(req, newparent), newname);

	fuse_reply_err(req, res == -1 ? errno : 0);
}

// Type: write
static void lo_unlink(fuse_req_t req, fuse_ino_t parent, const char *name)
{
	int res;

	res = unlinkat(lo_fd(req, parent), name, 0);

	fuse_reply_err(req, res == -1 ? errno : 0);
}

// David: decrease inode count by n, closing the fd and freeing the custom struct if new count == 0
// Type: write (reduces inode refcount)
static void unref_inode(struct lo_data *lo, struct lo_inode *inode, uint64_t n)
{
	if (!inode)
		return;

	pthread_mutex_lock(&lo->mutex);
	assert(inode->refcount >= n);
	inode->refcount -= n;
	if (!inode->refcount) {
		struct lo_inode *prev, *next;

		prev = inode->prev;
		next = inode->next;
		next->prev = prev;
		prev->next = next;

		pthread_mutex_unlock(&lo->mutex);
		close(inode->fd);
		free(inode);

	} else {
		pthread_mutex_unlock(&lo->mutex);
	}
}

// Type: write (uses unref_inode)
static void lo_forget_one(fuse_req_t req, fuse_ino_t ino, uint64_t nlookup)
{
	struct lo_data *lo = lo_data(req);
	struct lo_inode *inode = lo_inode(req, ino);

	if (lo_debug(req)) {
		fuse_log(FUSE_LOG_DEBUG, "  forget %lli %lli -%lli\n",
			(unsigned long long) ino,
			(unsigned long long) inode->refcount,
			(unsigned long long) nlookup);
	}

	unref_inode(lo, inode, nlookup);
}

// Type: write (uses lo_forget_one)
static void lo_forget(fuse_req_t req, fuse_ino_t ino, uint64_t nlookup)
{
	lo_forget_one(req, ino, nlookup);
	fuse_reply_none(req);
}

// Type: write (uses lo_forget_one)
static void lo_forget_multi(fuse_req_t req, size_t count,
				struct fuse_forget_data *forgets)
{
	int i;

	for (i = 0; i < count; i++)
		lo_forget_one(req, forgets[i].ino, forgets[i].nlookup);
	fuse_reply_none(req);
}

// David: read symbolic link and end the string with a null char
// Type: read
static void lo_readlink(fuse_req_t req, fuse_ino_t ino)
{
	char buf[PATH_MAX + 1];
	int res;

	res = readlinkat(lo_fd(req, ino), "", buf, sizeof(buf));
	if (res == -1)
		return (void) fuse_reply_err(req, errno);

	if (res == sizeof(buf))
		return (void) fuse_reply_err(req, ENAMETOOLONG);

	buf[res] = '\0';

	fuse_reply_readlink(req, buf);
}

struct lo_dirp {
	DIR *dp;
	struct dirent *entry;
	off_t offset;
};

// David: interpret file descriptors for directories as pointers to a custom directory struct
static struct lo_dirp *lo_dirp(struct fuse_file_info *fi)
{
	return (struct lo_dirp *) (uintptr_t) fi->fh;
}

// Type: write (opens directory, creates custom directory instance)
static void lo_opendir(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
	int error = ENOMEM;
	struct lo_data *lo = lo_data(req);
	struct lo_dirp *d;
	int fd;

	d = calloc(1, sizeof(struct lo_dirp));
	if (d == NULL)
		goto out_err;

	// David: get fd of directory
	fd = openat(lo_fd(req, ino), ".", O_RDONLY);
	if (fd == -1)
		goto out_errno;

	// David: turn directory fd into a DIR object and store inside d
	d->dp = fdopendir(fd);
	if (d->dp == NULL)
		goto out_errno;

	d->offset = 0;
	d->entry = NULL;

	// David: set file descriptor as the directory object d
	fi->fh = (uintptr_t) d;
	if (lo->cache == CACHE_ALWAYS)
		fi->cache_readdir = 1;
	fuse_reply_open(req, fi);
	return;

out_errno:
	error = errno;
out_err:
	if (d) {
		if (fd != -1)
			close(fd);
		free(d);
	}
	fuse_reply_err(req, error);
}

static int is_dot_or_dotdot(const char *name)
{
	return name[0] == '.' && (name[1] == '\0' ||
				  (name[1] == '.' && name[2] == '\0'));
}

// David: return all contents in a directory (like calling ls). Plus = 0 if lookup count of directory entries should not increase, 1 if they should.
// Type: read (although it increments dir entry, dir offset, and inode ref counts, this info is lost on restart)
static void lo_do_readdir(fuse_req_t req, fuse_ino_t ino, size_t size,
			  off_t offset, struct fuse_file_info *fi, int plus)
{
	struct lo_dirp *d = lo_dirp(fi);
	char *buf;
	char *p;
	size_t rem = size;
	int err;

	(void) ino;

	buf = calloc(1, size);
	if (!buf) {
		err = ENOMEM;
		goto error;
	}
	p = buf;

	// David: start reading the directory from the offset file
	if (offset != d->offset) {
		seekdir(d->dp, offset);
		d->entry = NULL;
		d->offset = offset;
	}
	while (1) {
		size_t entsize;
		off_t nextoff;
		const char *name;

		// David: if the directory doesn't have info about its first file, try to find it with readdir
		if (!d->entry) {
			errno = 0;
			d->entry = readdir(d->dp);
			if (!d->entry) {
				if (errno) {  // Error
					err = errno;
					goto error;
				} else {  // End of stream
					break; 
				}
			}
		}
		nextoff = d->entry->d_off;
		name = d->entry->d_name;
		fuse_ino_t entry_ino = 0;
		if (plus) {
			struct fuse_entry_param e;
			// David: only call do_lookup on actual directory entries, not . or ..
			if (is_dot_or_dotdot(name)) {
				e = (struct fuse_entry_param) {
					.attr.st_ino = d->entry->d_ino,
					.attr.st_mode = d->entry->d_type << 12,
				};
			} else {
				// David: do_lookup calls lo_find which increments refcount
				err = lo_do_lookup(req, ino, name, &e);
				if (err)
					goto error;
				entry_ino = e.ino;
			}

			entsize = fuse_add_direntry_plus(req, p, rem, name,
							 &e, nextoff);
		} else {
			// David: plus is false, don't call do_lookup
			struct stat st = {
				.st_ino = d->entry->d_ino,
				.st_mode = d->entry->d_type << 12,
			};
			entsize = fuse_add_direntry(req, p, rem, name,
						    &st, nextoff);
		}
		// David: stop if we're out of memory for the response
		if (entsize > rem) {
			if (entry_ino != 0) 
				lo_forget_one(req, entry_ino, 1);
			break;
		}
		
		p += entsize;
		rem -= entsize;

		d->entry = NULL;
		d->offset = nextoff;
	}

    err = 0;
error:
    // If there's an error, we can only signal it if we haven't stored
    // any entries yet - otherwise we'd end up with wrong lookup
    // counts for the entries that are already in the buffer. So we
    // return what we've collected until that point.
    if (err && rem == size)
	    fuse_reply_err(req, err);
    else
	    fuse_reply_buf(req, buf, size - rem);
    free(buf);
}

// Type: read
static void lo_readdir(fuse_req_t req, fuse_ino_t ino, size_t size,
		       off_t offset, struct fuse_file_info *fi)
{
	lo_do_readdir(req, ino, size, offset, fi, 0);
}

// Type: read
static void lo_readdirplus(fuse_req_t req, fuse_ino_t ino, size_t size,
			   off_t offset, struct fuse_file_info *fi)
{
	lo_do_readdir(req, ino, size, offset, fi, 1);
}

// Type: write
static void lo_releasedir(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
	struct lo_dirp *d = lo_dirp(fi);
	(void) ino;
	closedir(d->dp);
	free(d);
	fuse_reply_err(req, 0);
}

// Type: write
static void lo_create(fuse_req_t req, fuse_ino_t parent, const char *name,
		      mode_t mode, struct fuse_file_info *fi)
{
	int fd;
	struct lo_data *lo = lo_data(req);
	struct fuse_entry_param e;
	int err;

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "lo_create(parent=%" PRIu64 ", name=%s)\n",
			parent, name);

	// David: create the file with O_CREAT
	fd = openat(lo_fd(req, parent), name,
		    (fi->flags | O_CREAT) & ~O_NOFOLLOW, mode);
	if (fd == -1)
		return (void) fuse_reply_err(req, errno);

	// David: store the fd
	fi->fh = fd;
	if (lo->cache == CACHE_NEVER)
		fi->direct_io = 1;
	else if (lo->cache == CACHE_ALWAYS)
		fi->keep_cache = 1;

	// David: create the inode through do_lookup
	err = lo_do_lookup(req, parent, name, &e);
	if (err)
		fuse_reply_err(req, err);
	else
		fuse_reply_create(req, &e, fi);
}

// Type: fsync
static void lo_fsyncdir(fuse_req_t req, fuse_ino_t ino, int datasync,
			struct fuse_file_info *fi)
{
	int res;
	int fd = dirfd(lo_dirp(fi)->dp);
	(void) ino;
	if (datasync)
		res = fdatasync(fd);
	else
		res = fsync(fd);
	fuse_reply_err(req, res == -1 ? errno : 0);
}

// Type: write
static void lo_open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
	int fd;
	char buf[64];
	struct lo_data *lo = lo_data(req);

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "lo_open(ino=%" PRIu64 ", flags=%d)\n",
			ino, fi->flags);

	/* With writeback cache, kernel may send read requests even
	   when userspace opened write-only */
	if (lo->writeback && (fi->flags & O_ACCMODE) == O_WRONLY) {
		fi->flags &= ~O_ACCMODE;
		fi->flags |= O_RDWR;
	}

	/* With writeback cache, O_APPEND is handled by the kernel.
	   This breaks atomicity (since the file may change in the
	   underlying filesystem, so that the kernel's idea of the
	   end of the file isn't accurate anymore). In this example,
	   we just accept that. A more rigorous filesystem may want
	   to return an error here */
	if (lo->writeback && (fi->flags & O_APPEND))
		fi->flags &= ~O_APPEND;

	// David: use inode fd to create a file name, then open that and retrieve another fd?
	sprintf(buf, "/proc/self/fd/%i", lo_fd(req, ino));
	fd = open(buf, fi->flags & ~O_NOFOLLOW);
	if (fd == -1)
		return (void) fuse_reply_err(req, errno);

	fi->fh = fd;
	if (lo->cache == CACHE_NEVER)
		fi->direct_io = 1;
	else if (lo->cache == CACHE_ALWAYS)
		fi->keep_cache = 1;
	fuse_reply_open(req, fi);
}

// Type: write
static void lo_release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
	(void) ino;

	close(fi->fh);
	fuse_reply_err(req, 0);
}

// Type: read (doesn't prompt remote to do anything)
static void lo_flush(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
	int res;
	(void) ino;
	res = close(dup(fi->fh));
	fuse_reply_err(req, res == -1 ? errno : 0);
}

// Type: fsync
static void lo_fsync(fuse_req_t req, fuse_ino_t ino, int datasync,
		     struct fuse_file_info *fi)
{
	int res;
	(void) ino;
	if (datasync)
		res = fdatasync(fi->fh);
	else
		res = fsync(fi->fh);
	fuse_reply_err(req, res == -1 ? errno : 0);
}

// Type: read
static void lo_read(fuse_req_t req, fuse_ino_t ino, size_t size,
		    off_t offset, struct fuse_file_info *fi)
{
	struct fuse_bufvec buf = FUSE_BUFVEC_INIT(size);

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "lo_read(ino=%" PRIu64 ", size=%zd, "
			"off=%lu)\n", ino, size, (unsigned long) offset);

	buf.buf[0].flags = FUSE_BUF_IS_FD | FUSE_BUF_FD_SEEK;
	buf.buf[0].fd = fi->fh;
	buf.buf[0].pos = offset;

	fuse_reply_data(req, &buf, FUSE_BUF_SPLICE_MOVE);
}

// Type: write
static void lo_write_buf(fuse_req_t req, fuse_ino_t ino,
			 struct fuse_bufvec *in_buf, off_t off,
			 struct fuse_file_info *fi)
{
	(void) ino;
	ssize_t res;
	struct fuse_bufvec out_buf = FUSE_BUFVEC_INIT(fuse_buf_size(in_buf));

	out_buf.buf[0].flags = FUSE_BUF_IS_FD | FUSE_BUF_FD_SEEK;
	out_buf.buf[0].fd = fi->fh;
	out_buf.buf[0].pos = off;

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "lo_write(ino=%" PRIu64 ", size=%zd, off=%lu)\n",
			ino, out_buf.buf[0].size, (unsigned long) off);

	res = fuse_buf_copy(&out_buf, in_buf, 0);
	if(res < 0)
		fuse_reply_err(req, -res);
	else
		fuse_reply_write(req, (size_t) res);
}

// Type: read
static void lo_statfs(fuse_req_t req, fuse_ino_t ino)
{
	int res;
	struct statvfs stbuf;

	res = fstatvfs(lo_fd(req, ino), &stbuf);
	if (res == -1)
		fuse_reply_err(req, errno);
	else
		fuse_reply_statfs(req, &stbuf);
}

// Type: write
static void lo_fallocate(fuse_req_t req, fuse_ino_t ino, int mode,
			 off_t offset, off_t length, struct fuse_file_info *fi)
{
	int err = EOPNOTSUPP;
	(void) ino;

#ifdef HAVE_FALLOCATE
	err = fallocate(fi->fh, mode, offset, length);
	if (err < 0)
		err = errno;

#elif defined(HAVE_POSIX_FALLOCATE)
	if (mode) {
		fuse_reply_err(req, EOPNOTSUPP);
		return;
	}

	err = posix_fallocate(fi->fh, offset, length);
#endif

	fuse_reply_err(req, err);
}

// Type: read (locks are not persisted after restart so we don't care)
static void lo_flock(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi,
		     int op)
{
	int res;
	(void) ino;

	res = flock(fi->fh, op);

	fuse_reply_err(req, res == -1 ? errno : 0);
}

// Type: read
static void lo_getxattr(fuse_req_t req, fuse_ino_t ino, const char *name,
			size_t size)
{
	char *value = NULL;
	char procname[64];
	struct lo_inode *inode = lo_inode(req, ino);
	ssize_t ret;
	int saverr;

	saverr = ENOSYS;
	if (!lo_data(req)->xattr)
		goto out;

	if (lo_debug(req)) {
		fuse_log(FUSE_LOG_DEBUG, "lo_getxattr(ino=%" PRIu64 ", name=%s size=%zd)\n",
			ino, name, size);
	}

	sprintf(procname, "/proc/self/fd/%i", inode->fd);

	// David: non-zero size = reply with fuse_reply_buf
	if (size) {
		value = malloc(size);
		if (!value)
			goto out_err;

		ret = getxattr(procname, name, value, size);
		if (ret == -1)
			goto out_err;
		saverr = 0;
		if (ret == 0)
			goto out;

		fuse_reply_buf(req, value, ret);
	} else {
		ret = getxattr(procname, name, NULL, 0);
		if (ret == -1)
			goto out_err;

		fuse_reply_xattr(req, ret);
	}
out_free:
	free(value);
	return;

out_err:
	saverr = errno;
out:
	fuse_reply_err(req, saverr);
	goto out_free;
}

// Type: read
static void lo_listxattr(fuse_req_t req, fuse_ino_t ino, size_t size)
{
	char *value = NULL;
	char procname[64];
	struct lo_inode *inode = lo_inode(req, ino);
	ssize_t ret;
	int saverr;

	saverr = ENOSYS;
	if (!lo_data(req)->xattr)
		goto out;

	if (lo_debug(req)) {
		fuse_log(FUSE_LOG_DEBUG, "lo_listxattr(ino=%" PRIu64 ", size=%zd)\n",
			ino, size);
	}

	sprintf(procname, "/proc/self/fd/%i", inode->fd);

	// David: non-zero size = reply with fuse_reply_buf
	if (size) {
		value = malloc(size);
		if (!value)
			goto out_err;

		ret = listxattr(procname, value, size);
		if (ret == -1)
			goto out_err;
		saverr = 0;
		if (ret == 0)
			goto out;

		fuse_reply_buf(req, value, ret);
	} else {
		ret = listxattr(procname, NULL, 0);
		if (ret == -1)
			goto out_err;

		fuse_reply_xattr(req, ret);
	}
out_free:
	free(value);
	return;

out_err:
	saverr = errno;
out:
	fuse_reply_err(req, saverr);
	goto out_free;
}

// Type: write
static void lo_setxattr(fuse_req_t req, fuse_ino_t ino, const char *name,
			const char *value, size_t size, int flags)
{
	char procname[64];
	struct lo_inode *inode = lo_inode(req, ino);
	ssize_t ret;
	int saverr;

	saverr = ENOSYS;
	if (!lo_data(req)->xattr)
		goto out;

	if (lo_debug(req)) {
		fuse_log(FUSE_LOG_DEBUG, "lo_setxattr(ino=%" PRIu64 ", name=%s value=%s size=%zd)\n",
			ino, name, value, size);
	}

	sprintf(procname, "/proc/self/fd/%i", inode->fd);

	ret = setxattr(procname, name, value, size, flags);
	saverr = ret == -1 ? errno : 0;

out:
	fuse_reply_err(req, saverr);
}

// Type: write
static void lo_removexattr(fuse_req_t req, fuse_ino_t ino, const char *name)
{
	char procname[64];
	struct lo_inode *inode = lo_inode(req, ino);
	ssize_t ret;
	int saverr;

	saverr = ENOSYS;
	if (!lo_data(req)->xattr)
		goto out;

	if (lo_debug(req)) {
		fuse_log(FUSE_LOG_DEBUG, "lo_removexattr(ino=%" PRIu64 ", name=%s)\n",
			ino, name);
	}

	sprintf(procname, "/proc/self/fd/%i", inode->fd);

	ret = removexattr(procname, name);
	saverr = ret == -1 ? errno : 0;

out:
	fuse_reply_err(req, saverr);
}

// Type: write
#ifdef HAVE_COPY_FILE_RANGE
static void lo_copy_file_range(fuse_req_t req, fuse_ino_t ino_in, off_t off_in,
			       struct fuse_file_info *fi_in,
			       fuse_ino_t ino_out, off_t off_out,
			       struct fuse_file_info *fi_out, size_t len,
			       int flags)
{
	ssize_t res;

	if (lo_debug(req))
		fuse_log(FUSE_LOG_DEBUG, "lo_copy_file_range(ino=%" PRIu64 "/fd=%lu, "
				"off=%lu, ino=%" PRIu64 "/fd=%lu, "
				"off=%lu, size=%zd, flags=0x%x)\n",
			ino_in, fi_in->fh, off_in, ino_out, fi_out->fh, off_out,
			len, flags);

	res = copy_file_range(fi_in->fh, &off_in, fi_out->fh, &off_out, len,
			      flags);
	if (res < 0)
		fuse_reply_err(req, errno);
	else
		fuse_reply_write(req, res);
}
#endif

// Type: write (changes file offset)
static void lo_lseek(fuse_req_t req, fuse_ino_t ino, off_t off, int whence,
		     struct fuse_file_info *fi)
{
	off_t res;

	(void)ino;
	res = lseek(fi->fh, off, whence);
	if (res != -1)
		fuse_reply_lseek(req, res);
	else
		fuse_reply_err(req, errno);
}

static const struct fuse_lowlevel_ops lo_oper = {
	.init		= lo_init,
	.destroy	= lo_destroy,
	.lookup		= lo_lookup,
	.mkdir		= lo_mkdir,
	.mknod		= lo_mknod,
	.symlink	= lo_symlink,
	.link		= lo_link,
	.unlink		= lo_unlink,
	.rmdir		= lo_rmdir,
	.rename		= lo_rename,
	.forget		= lo_forget,
	.forget_multi	= lo_forget_multi,
	.getattr	= lo_getattr,
	.setattr	= lo_setattr,
	.readlink	= lo_readlink,
	.opendir	= lo_opendir,
	.readdir	= lo_readdir,
	.readdirplus	= lo_readdirplus,
	.releasedir	= lo_releasedir,
	.fsyncdir	= lo_fsyncdir,
	.create		= lo_create,
	.open		= lo_open,
	.release	= lo_release,
	.flush		= lo_flush,
	.fsync		= lo_fsync,
	.read		= lo_read,
	.write_buf      = lo_write_buf,
	.statfs		= lo_statfs,
	.fallocate	= lo_fallocate,
	.flock		= lo_flock,
	.getxattr	= lo_getxattr,
	.listxattr	= lo_listxattr,
	.setxattr	= lo_setxattr,
	.removexattr	= lo_removexattr,
#ifdef HAVE_COPY_FILE_RANGE
	.copy_file_range = lo_copy_file_range,
#endif
	.lseek		= lo_lseek,
};

int main(int argc, char *argv[])
{
	struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
	struct fuse_session *se;
	struct fuse_cmdline_opts opts;
	struct fuse_loop_config config;
	struct lo_data lo = { .debug = 0,
	                      .writeback = 0 };
	int ret = -1;

	/* Don't mask creation mode, kernel already did that */
	umask(0);

	pthread_mutex_init(&lo.mutex, NULL);
	lo.root.next = lo.root.prev = &lo.root;
	lo.root.fd = -1;
	lo.cache = CACHE_NORMAL;

	if (fuse_parse_cmdline(&args, &opts) != 0)
		return 1;
	if (opts.show_help) {
		printf("usage: %s [options] <mountpoint>\n\n", argv[0]);
		fuse_cmdline_help();
		fuse_lowlevel_help();
		passthrough_ll_help();
		ret = 0;
		goto err_out1;
	} else if (opts.show_version) {
		printf("FUSE library version %s\n", fuse_pkgversion());
		fuse_lowlevel_version();
		ret = 0;
		goto err_out1;
	}

	if(opts.mountpoint == NULL) {
		printf("usage: %s [options] <mountpoint>\n", argv[0]);
		printf("       %s --help\n", argv[0]);
		ret = 1;
		goto err_out1;
	}

	if (fuse_opt_parse(&args, &lo, lo_opts, NULL)== -1)
		return 1;

	lo.debug = opts.debug;
	lo.root.refcount = 2;
	if (lo.source) {
		struct stat stat;
		int res;

		res = lstat(lo.source, &stat);
		if (res == -1) {
			fuse_log(FUSE_LOG_ERR, "failed to stat source (\"%s\"): %m\n",
				 lo.source);
			exit(1);
		}
		if (!S_ISDIR(stat.st_mode)) {
			fuse_log(FUSE_LOG_ERR, "source is not a directory\n");
			exit(1);
		}

	} else {
		lo.source = strdup("/");
		if(!lo.source) {
			fuse_log(FUSE_LOG_ERR, "fuse: memory allocation failed\n");
			exit(1);
		}
	}
	if (!lo.timeout_set) {
		switch (lo.cache) {
		case CACHE_NEVER:
			lo.timeout = 0.0;
			break;

		case CACHE_NORMAL:
			lo.timeout = 1.0;
			break;

		case CACHE_ALWAYS:
			lo.timeout = 86400.0;
			break;
		}
	} else if (lo.timeout < 0) {
		fuse_log(FUSE_LOG_ERR, "timeout is negative (%lf)\n",
			 lo.timeout);
		exit(1);
	}

	lo.root.fd = open(lo.source, O_PATH);
	if (lo.root.fd == -1) {
		fuse_log(FUSE_LOG_ERR, "open(\"%s\", O_PATH): %m\n",
			 lo.source);
		exit(1);
	}

	se = fuse_session_new(&args, &lo_oper, sizeof(lo_oper), &lo);
	if (se == NULL)
	    goto err_out1;

	if (fuse_set_signal_handlers(se) != 0)
	    goto err_out2;

	if (fuse_session_mount(se, opts.mountpoint) != 0)
	    goto err_out3;

	fuse_daemonize(opts.foreground);

	/* Block until ctrl+c or fusermount -u */
	if (opts.singlethread)
		ret = fuse_session_loop(se);
	else {
		config.clone_fd = opts.clone_fd;
		config.max_idle_threads = opts.max_idle_threads;
		ret = fuse_session_loop_mt(se, &config);
	}

	fuse_session_unmount(se);
err_out3:
	fuse_remove_signal_handlers(se);
err_out2:
	fuse_session_destroy(se);
err_out1:
	free(opts.mountpoint);
	fuse_opt_free_args(&args);

	if (lo.root.fd >= 0)
		close(lo.root.fd);

	free(lo.source);
	return ret ? 1 : 0;
}
